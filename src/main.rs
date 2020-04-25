use std::ffi::CString;
use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    descriptor::{
        descriptor_set::PersistentDescriptorSet, pipeline_layout::PipelineLayoutAbstract,
    },
    device::{Device, DeviceExtensions, Queue},
    format::ClearValue,
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::SwapchainImage,
    instance::{Instance, PhysicalDevice, RawInstanceExtensions},
    pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract},
    swapchain::{self, FullscreenExclusive, PresentMode, Surface, SurfaceTransform, Swapchain},
    sync::{self, GpuFuture},
    VulkanObject,
};

use sdl2::{event::Event, video::Window, Sdl};

use fragile::Fragile;

mod shaders;

static DIMS: [u32; 2] = [600, 600];

#[derive(Default, Copy, Clone)]
struct Vertex {
    a_position: [f32; 2],
    a_color: [f32; 4],
}

impl Vertex {
    pub fn new(xy: [f32; 2], rgba: [f32; 4]) -> Self {
        Vertex {
            a_position: xy,
            a_color: rgba,
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
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    dyn_state: DynamicState,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
}

fn required_extensions(window: &Window) -> RawInstanceExtensions {
    let ext_names: Vec<&str> = window.vulkan_instance_extensions().unwrap();
    let ext_strs = ext_names
        .into_iter()
        .map(|s| CString::new(s.as_bytes()).unwrap());
    RawInstanceExtensions::new(ext_strs)
}

fn make_viewport(w: u32, h: u32) -> Viewport {
    Viewport {
        origin: [0.0, 0.0],
        // I have no idea why the API wants floats here
        dimensions: [w as f32, h as f32],
        depth_range: 0.0..1.0,
    }
}

impl Game {
    fn make_framebuffers(
        images: Vec<Arc<SwapchainImage<Fragile<Window>>>>,
        render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        let mut fbs: Vec<Arc<dyn FramebufferAbstract + Send + Sync>> = Vec::new();
        for image in images {
            let fb = Framebuffer::start(render_pass.clone())
                .add(image.clone())
                .unwrap()
                .build()
                .unwrap();
            fbs.push(Arc::new(fb));
        }
        fbs
    }

    pub fn rebuild_swapchain(&mut self) {
        let surface = self.swapchain.surface();
        let window = surface.window().get();
        let (w, h) = window.vulkan_drawable_size();

        let (new_sc, new_imgs) = self.swapchain.recreate_with_dimensions([w, h]).unwrap();
        self.swapchain = new_sc;
        self.framebuffers = Self::make_framebuffers(new_imgs, self.render_pass.clone());
        self.dyn_state.viewports.replace(vec![make_viewport(w, h)]);
    }

    pub fn new() -> Self {
        let sdl = sdl2::init().unwrap();
        let video_subsystem = sdl.video().unwrap();

        let window = video_subsystem
            .window("stupid horse", DIMS[0], DIMS[1])
            .resizable()
            .vulkan()
            .build()
            .unwrap();

        let exts = required_extensions(&window);

        let inst = Instance::new(None, exts, None).expect("failed to create instance");

        let phys_gpu = PhysicalDevice::enumerate(&inst)
            .next()
            .expect("no device available");

        let (gpu, mut queues) = {
            let queue_family = phys_gpu
                .queue_families()
                .find(|&q| q.supports_graphics())
                .expect("couldn't find a graphical queue family");

            let device_ext = DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::none()
            };

            Device::new(
                phys_gpu,
                phys_gpu.supported_features(),
                &device_ext,
                [(queue_family, 0.5)].iter().cloned(),
            )
            .expect("failed to create device")
        };
        let queue = queues.next().unwrap();

        let surface = unsafe {
            let raw_instance = inst.internal_object();
            let raw_surface = window.vulkan_create_surface(raw_instance).unwrap();
            Arc::new(Surface::from_raw_surface(
                inst.clone(),
                raw_surface,
                Fragile::new(window),
            ))
        };

        let caps = surface.capabilities(phys_gpu).unwrap();

        let (swapchain, images) = Swapchain::new(
            gpu.clone(),
            surface.clone(),
            caps.min_image_count,
            caps.supported_formats[0].0,
            caps.current_extent.unwrap_or(DIMS),
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

        let render_pass = {
            let pass = vulkano::single_pass_renderpass!(
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
            .unwrap();
            Arc::new(pass)
        };

        let framebuffers = Self::make_framebuffers(images, render_pass.clone());

        let pipeline = {
            let vs = shaders::bg::vert::Shader::load(gpu.clone()).unwrap();
            let fs = shaders::bg::frag::Shader::load(gpu.clone()).unwrap();
            let obj = GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(gpu.clone())
                .unwrap();
            Arc::new(obj)
        };

        // No longer treating the DIMS const as authoritative
        let dims = swapchain.dimensions();

        let dyn_state = DynamicState {
            viewports: Some(vec![make_viewport(dims[0], dims[1])]),
            ..Default::default()
        };

        Self {
            sdl: sdl,
            gpu: gpu,
            queue: queue,
            swapchain: swapchain,
            framebuffers: framebuffers,
            pipeline: pipeline,
            dyn_state: dyn_state,
            render_pass: render_pass,
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

    let mut prev_frame_end = Some(Box::new(sync::now(game.gpu.clone())) as Box<dyn GpuFuture>);

    let mut event_pump = game.sdl.event_pump().unwrap();

    let bg_verts = {
        let topleft = [-1.0, -1.0];
        let topright = [1.0, -1.0];
        let bottomleft = [-1.0, 1.0];
        let bottomright = [1.0, 1.0];
        let center = [0.0, 0.0];

        let black = [0.0, 0.0, 0.0, 1.0];
        let gray1 = [0.25, 0.25, 0.25, 1.0];
        let gray2 = [0.5, 0.5, 0.5, 1.0];
        let gray3 = [0.75, 0.75, 0.75, 1.0];
        let white = [1.0, 1.0, 1.0, 1.0];

        let verts = tris![
            (topleft, topright, center, black),
            (topright, bottomright, center, gray1),
            (bottomright, bottomleft, center, gray2),
            (bottomleft, topleft, center, gray3),
            ([-0.125, -0.125], [0.125, -0.125], [0.0, 0.125], white),
        ];

        CpuAccessibleBuffer::from_data(game.gpu.clone(), BufferUsage::all(), false, verts).unwrap()
    };

    let (uniforms, uniforms_desc) = {
        let layout = game.pipeline.descriptor_set_layout(0).unwrap();
        let dims = game.swapchain.dimensions();

        let buf = CpuAccessibleBuffer::from_data(
            game.gpu.clone(),
            BufferUsage::all(),
            false,
            (
                // beware of alignment discrepancies between GLSL and Rust
                (0.0f32, 0.0f32),                 // click position
                (dims[0] as f32, dims[1] as f32), // window dimensions
                0.0f32,                           // time
            ),
        )
        .unwrap();

        let desc = PersistentDescriptorSet::start(layout.clone())
            .add_buffer(buf.clone())
            .unwrap()
            .build()
            .unwrap();

        (buf, Arc::new(desc))
    };

    let mut rebuild_swapchain = false;

    'running: loop {
        prev_frame_end.as_mut().unwrap().cleanup_finished();

        // no idea where in the event loop this should actually take place
        // no idea whether this even counts as an event loop
        if rebuild_swapchain {
            game.rebuild_swapchain();
            let dims = game.swapchain.dimensions();
            uniforms.write().unwrap().1 = (dims[0] as f32, dims[1] as f32);
            rebuild_swapchain = false;
        }

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::MouseMotion { .. } => (),
                Event::MouseButtonDown { x, y, .. } => {
                    let dims = game.swapchain.dimensions();
                    let (w, h) = (dims[0], dims[1]);
                    uniforms.write().unwrap().0 = (
                        (2 * x - w as i32) as f32 / w as f32,
                        (2 * y - h as i32) as f32 / h as f32,
                    );
                }
                _ => println!("{:?}", event),
            }
        }

        uniforms.write().unwrap().2 += 0.1;

        let (image_num, suboptimal, acquire_future) =
            swapchain::acquire_next_image(game.swapchain.clone(), None).unwrap();

        if suboptimal {
            rebuild_swapchain = true;
        }

        let fb = game.framebuffers[image_num].clone();

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            game.gpu.clone(),
            game.queue.family(),
        )
        .unwrap()
        .begin_render_pass(fb, false, vec![ClearValue::Float([0.0, 0.0, 1.0, 1.0])])
        .unwrap()
        .draw(
            game.pipeline.clone(),
            &game.dyn_state,
            vec![bg_verts.clone()],
            uniforms_desc.clone(),
            (),
        )
        .expect("draw call failed")
        .end_render_pass()
        .unwrap()
        .build()
        .unwrap();

        let future = prev_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(game.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(game.queue.clone(), game.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                prev_frame_end.replace(Box::new(future));
            }
            Err(e) => {
                panic!("{:?}", e);
            }
        }
        std::thread::sleep(std::time::Duration::new(0, 1_000_000_000u32 / 60));
    }
}
