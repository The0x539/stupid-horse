use fragile::Fragile;
use sdl2::{event::Event, video::Window, Sdl};
use std::ffi::CString;
use std::sync::{Arc, Mutex};
use vulkano::{
    buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    descriptor::descriptor_set::FixedSizeDescriptorSetsPool,
    device::{Device, DeviceExtensions, Queue},
    format::ClearValue,
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::SwapchainImage,
    instance::{Instance, PhysicalDevice, RawInstanceExtensions},
    pipeline::{
        vertex::Vertex as VertexAbstract, viewport::Viewport, GraphicsPipeline,
        GraphicsPipelineAbstract,
    },
    swapchain::{self, FullscreenExclusive, PresentMode, Surface, SurfaceTransform, Swapchain},
    sync::{self, GpuFuture},
    VulkanObject,
};

mod shaders;
use shaders::{
    bg::{frag::Shader as bg_fs, vert::Shader as bg_vs},
    fg::{frag::Shader as fg_fs, vert::Shader as fg_vs},
    ShaderAbstract,
};

static WIN_WIDTH: u32 = 540;
static WIN_HEIGHT: u32 = 540;
static ASPECT_RATIO: f32 = WIN_WIDTH as f32 / WIN_HEIGHT as f32;

#[derive(Default, Copy, Clone)]
struct Vertex {
    a_position: [f32; 2],
    a_color: [f32; 3],
}

impl Vertex {
    pub fn new(xy: [f32; 2], rgb: [f32; 3]) -> Self {
        Vertex {
            a_position: xy,
            a_color: rgb,
        }
    }
}

vulkano::impl_vertex!(Vertex, a_position, a_color);

struct Game {
    sdl: Sdl,
    gpu: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Fragile<Window>>>,
    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    dyn_state: DynamicState,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    future: Option<Box<dyn GpuFuture>>,
}

fn required_extensions(window: &Window) -> RawInstanceExtensions {
    window
        .vulkan_instance_extensions()
        .unwrap()
        .into_iter()
        .map(|s| CString::new(s.as_bytes()).unwrap())
        .collect()
}

impl Game {
    fn make_framebuffers(
        images: Vec<Arc<SwapchainImage<Fragile<Window>>>>,
        render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        images
            .into_iter()
            .map(|image| {
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap()
            })
            .map(|fb| Arc::new(fb) as Arc<dyn FramebufferAbstract + Send + Sync>)
            .collect()
    }

    fn make_viewport([w, h]: [u32; 2]) -> Viewport {
        let (win_w, win_h) = (w as f32, h as f32);
        let vp_w = win_w.min(win_h * ASPECT_RATIO);
        let vp_h = win_h.min(win_w / ASPECT_RATIO);
        Viewport {
            origin: [(win_w - vp_w) / 2.0, (win_h - vp_h) / 2.0],
            dimensions: [vp_w, vp_h],
            depth_range: 0.0..1.0,
        }
    }

    pub fn get_viewport(&self) -> Viewport {
        self.dyn_state.viewports.as_ref().unwrap()[0].clone()
    }

    pub fn rebuild_swapchain(&mut self) {
        let surface = self.swapchain.surface();
        let window = surface.window().get();
        let (w, h) = window.vulkan_drawable_size();

        let (new_sc, new_imgs) = self.swapchain.recreate_with_dimensions([w, h]).unwrap();
        self.swapchain = new_sc;
        self.framebuffers = Self::make_framebuffers(new_imgs, self.render_pass.clone());
        self.dyn_state.viewports.as_mut().unwrap()[0] = Self::make_viewport([w, h]);
    }

    pub fn make_pipeline<Vs: ShaderAbstract, Fs: ShaderAbstract, V: VertexAbstract>(
        &self,
    ) -> Arc<impl GraphicsPipelineAbstract> {
        let vs = <Vs>::load(self.gpu.clone()).unwrap();
        let fs = <Fs>::load(self.gpu.clone()).unwrap();
        let pipeline = GraphicsPipeline::start()
            .vertex_input_single_buffer::<V>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(self.render_pass.clone(), 0).unwrap())
            .build(self.gpu.clone())
            .unwrap();

        Arc::new(pipeline)
    }

    pub fn make_uniforms<T>(
        &self,
        pipeline: impl GraphicsPipelineAbstract,
        idx: usize,
    ) -> (CpuBufferPool<T>, Mutex<FixedSizeDescriptorSetsPool>) {
        let layout = pipeline.descriptor_set_layout(idx).unwrap();
        let buf_pool = CpuBufferPool::uniform_buffer(self.gpu.clone());
        let desc_pool = FixedSizeDescriptorSetsPool::new(layout.clone());
        (buf_pool, Mutex::new(desc_pool))
    }

    pub fn make_immutable_buffer<T: Send + Sync + 'static>(
        &mut self,
        verts: T,
    ) -> Arc<ImmutableBuffer<T>> {
        let (buf, fut) =
            ImmutableBuffer::from_data(verts, BufferUsage::all(), self.queue.clone()).unwrap();
        self.future = self
            .future
            .take()
            .map(|x| Box::new(x.join(fut)) as Box<dyn GpuFuture>);
        buf
    }

    pub fn new() -> Self {
        let sdl = sdl2::init().unwrap();
        let video_subsystem = sdl.video().unwrap();
        let window = video_subsystem
            .window("stupid horse", WIN_WIDTH, WIN_HEIGHT)
            .resizable()
            .vulkan()
            .build()
            .unwrap();

        let exts = required_extensions(&window);
        let inst = Instance::new(None, exts, None).expect("failed to create instance");

        let (gpu, queue) = {
            let phys_gpu = PhysicalDevice::enumerate(&inst)
                .next()
                .expect("no device available");

            let queue_family = phys_gpu
                .queue_families()
                .find(|q| q.supports_graphics())
                .expect("couldn't find a graphical queue family");

            let (dev, mut queues) = Device::new(
                phys_gpu,
                phys_gpu.supported_features(),
                &DeviceExtensions {
                    khr_swapchain: true,
                    ..DeviceExtensions::none()
                },
                std::iter::once((queue_family, 0.5)),
            )
            .expect("failed to create device");
            (dev, queues.next().unwrap())
        };

        let surface = unsafe {
            let raw_instance = inst.internal_object();
            let raw_surface = window.vulkan_create_surface(raw_instance).unwrap();
            Arc::new(Surface::from_raw_surface(
                inst.clone(),
                raw_surface,
                Fragile::new(window),
            ))
        };

        let caps = surface.capabilities(gpu.physical_device()).unwrap();
        let (swapchain, images) = Swapchain::new(
            gpu.clone(),
            surface.clone(),
            caps.min_image_count,
            caps.supported_formats[0].0,
            [WIN_WIDTH, WIN_HEIGHT],
            1,
            caps.supported_usage_flags,
            &queue,
            SurfaceTransform::Identity,
            caps.supported_composite_alpha.iter().next().unwrap(),
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            caps.supported_formats[0].1,
        )
        .expect("failed to create swapchain");

        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(
                gpu.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    }
                },
                pass: {color: [color], depth_stencil: {}}
            )
            .unwrap(),
        );

        let framebuffers = Self::make_framebuffers(images, render_pass.clone());

        let dyn_state = DynamicState {
            viewports: Some(vec![Self::make_viewport(swapchain.dimensions())]),
            ..Default::default()
        };

        let future = Some(Box::new(sync::now(gpu.clone())) as Box<dyn GpuFuture>);

        Self {
            sdl,
            gpu,
            queue,
            swapchain,
            framebuffers,
            dyn_state,
            render_pass,
            future,
        }
    }
}

macro_rules! tris {
    ($(($v1:expr, $v2:expr, $v3:expr, $c:expr)),*$(,)*) => {
        [$(
            Vertex::new($v1, $c),
            Vertex::new($v2, $c),
            Vertex::new($v3, $c)
        ),*]
    }
}

fn main() {
    let mut game = Game::new();

    let mut event_pump = game.sdl.event_pump().unwrap();

    let bg_verts = {
        let topleft = [-1.0, -1.0];
        let topright = [1.0, -1.0];
        let bottomleft = [-1.0, 1.0];
        let bottomright = [1.0, 1.0];
        let center = [0.0, 0.0];

        let black = [0.0, 0.0, 0.0];
        let gray1 = [0.25, 0.25, 0.25];
        let gray2 = [0.5, 0.5, 0.5];
        let gray3 = [0.75, 0.75, 0.75];
        let white = [1.0, 1.0, 1.0];

        game.make_immutable_buffer(tris![
            (topleft, topright, center, black),
            (topright, bottomright, center, gray1),
            (bottomright, bottomleft, center, gray2),
            (bottomleft, topleft, center, gray3),
            ([-0.125, -0.125], [0.125, -0.125], [0.0, 0.125], white),
        ])
    };

    let fg_verts = {
        let red = [1.0, 0.0, 0.0];
        let green = [0.0, 1.0, 0.0];
        let blue = [0.0, 0.0, 1.0];

        // the six points in a triforce, starting at the top and going clockwise
        let p = (
            [0.0, 1.0],
            [f32::sqrt(3.0) / 4.0, 0.25],
            [f32::sqrt(3.0) / 2.0, -0.5],
            [0.0, -0.5],
            [-f32::sqrt(3.0) / 2.0, -0.5],
            [-f32::sqrt(3.0) / 4.0, 0.25],
        );

        game.make_immutable_buffer(tris![
            (p.0, p.1, p.5, red),
            (p.1, p.2, p.3, green),
            (p.5, p.3, p.4, blue),
        ])
    };

    let bg_pipeline = game.make_pipeline::<bg_vs, bg_fs, Vertex>();
    let fg_pipeline = game.make_pipeline::<fg_vs, fg_fs, Vertex>();

    let mut click_pos = (0.0f32, 0.0f32);
    let mut time = 0.0f32;
    let mut scale = 0.5f32;

    let (fg_buf_pool, mut fg_desc_pool) = game.make_uniforms(fg_pipeline.clone(), 0);

    let mut rebuild_swapchain = false;

    'running: loop {
        game.future.as_mut().unwrap().cleanup_finished();

        // no idea where in the event loop this should actually take place
        // no idea whether this even counts as an event loop
        if rebuild_swapchain {
            game.rebuild_swapchain();
            rebuild_swapchain = false;
        }

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::MouseMotion { .. } => (),
                Event::MouseButtonDown { x, y, .. } => {
                    let Viewport {
                        origin: [vp_x, vp_y],
                        dimensions: [vp_w, vp_h],
                        ..
                    } = game.get_viewport();

                    click_pos.0 = (x as f32 - vp_x) * 2.0 / vp_w - 1.0;
                    click_pos.1 = (y as f32 - vp_y) * 2.0 / vp_h - 1.0;
                }
                Event::MouseWheel { y, .. } => {
                    scale *= f32::powi(1.1, y);
                }
                _ => println!("{:?}", event),
            }
        }

        time += 0.1;

        let (image_num, suboptimal, acquire_future) =
            swapchain::acquire_next_image(game.swapchain.clone(), None).unwrap();

        if suboptimal {
            rebuild_swapchain = true;
        }

        let fb = game.framebuffers[image_num].clone();

        let uniforms = shaders::fg::Uniforms {
            click_pos,
            time,
            scale,
        };
        let buf = fg_buf_pool.next(uniforms).unwrap();
        let desc = fg_desc_pool
            .get_mut()
            .unwrap()
            .next()
            .add_buffer(buf)
            .unwrap()
            .build()
            .unwrap();

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            game.gpu.clone(),
            game.queue.family(),
        )
        .unwrap()
        .begin_render_pass(fb, false, vec![ClearValue::Float([0.0, 0.0, 1.0, 1.0])])
        .unwrap()
        .draw(
            bg_pipeline.clone(),
            &game.dyn_state,
            vec![bg_verts.clone()],
            (),
            (),
        )
        .expect("bg draw call failed")
        .draw(
            fg_pipeline.clone(),
            &game.dyn_state,
            vec![fg_verts.clone()],
            desc,
            (),
        )
        .expect("fg draw call failed")
        .end_render_pass()
        .unwrap()
        .build()
        .unwrap();

        let future = game
            .future
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(game.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(game.queue.clone(), game.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        game.future.replace(Box::new(future.unwrap()));

        std::thread::sleep(std::time::Duration::new(0, 1_000_000_000u32 / 60));
    }
}
