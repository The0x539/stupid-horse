use std::sync::Arc;
use std::ffi::CString;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    descriptor::{descriptor_set::PersistentDescriptorSet, pipeline_layout::PipelineLayoutAbstract},
    device::{Device, DeviceExtensions, Queue},
    format::ClearValue,
    framebuffer::{self as vk_fb, Subpass, RenderPassDesc},
    image as vk_img,
    instance::{Instance, PhysicalDevice, RawInstanceExtensions},
    pipeline::{viewport::Viewport, GraphicsPipeline},
    swapchain::{self, FullscreenExclusive, PresentMode, Surface, SurfaceTransform, Swapchain},
    sync::{self, GpuFuture},
    VulkanObject,
};

use sdl2::{Sdl, event::Event, video::Window};

use fragile::Fragile;

mod vs { vulkano_shaders::shader! { ty: "vertex", path: "src/shaders/vertex.glsl" } }
mod fs { vulkano_shaders::shader! { ty: "fragment", path: "src/shaders/fragment.glsl" } }

mod passdesc;
use passdesc::Desc;

static DIMS: [u32; 2] = [600, 600];

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: (f32, f32)
}

impl Vertex {
    pub fn new(x: f32, y: f32) -> Self {
        Vertex { position: (x, y) }
    }
}

vulkano::impl_vertex!(Vertex, position);

type RenderPass = vk_fb::RenderPass<Desc>;
type SwapchainImage = vk_img::SwapchainImage<Fragile<Window>>;
type FramebufferImage = ((), Arc<SwapchainImage>);
type Framebuffer = vk_fb::Framebuffer<Arc<RenderPass>, FramebufferImage>;

struct Game {
    sdl: Sdl,
    gpu: Arc<Device>,
    queue: Arc<Queue>,

    swapchain: Arc<Swapchain<Fragile<Window>>>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

fn required_extensions(window: &Window) -> RawInstanceExtensions {
    let ext_names: Vec<&str> = window.vulkan_instance_extensions().unwrap();

    let ext_strs = ext_names.into_iter().map(|s| CString::new(s.as_bytes()).unwrap());

    RawInstanceExtensions::new(ext_strs)
}

impl Game {
    pub fn new() -> Self {
        let sdl = sdl2::init().unwrap();
        let video_subsystem = sdl.video().unwrap();

        let window = video_subsystem
            .window("stupid horse", DIMS[0], DIMS[1])
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
            Arc::new(Surface::from_raw_surface(inst.clone(), raw_surface, Fragile::new(window)))
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
        ).expect("failed to create swapchain");

        let render_pass = {
            let pass = Desc{color: (swapchain.format(), 1)}.build_render_pass(gpu.clone()).unwrap();
            Arc::new(pass)
        };

        let framebuffers = images
            .iter()
            .map(|image| {
                Arc::new(
                    vk_fb::Framebuffer::start(render_pass.clone())
                        .add(image.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();

        Self {
            sdl: sdl,
            gpu: gpu,
            queue: queue,
            swapchain: swapchain,
            framebuffers: framebuffers,
        }
    }
}

fn main() {
    let game = Game::new();

    let mut prev_frame_end = Some(Box::new(sync::now(game.gpu.clone())) as Box<dyn GpuFuture>);

    let mut event_pump = game.sdl.event_pump().unwrap();

    let clear_values = [ClearValue::Float([0.0, 0.0, 1.0, 1.0])];
    let vs = vs::Shader::load(game.gpu.clone()).unwrap();
    let fs = fs::Shader::load(game.gpu.clone()).unwrap();
    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(game.framebuffers[0].render_pass().clone(), 0).unwrap())
            .build(game.gpu.clone())
            .unwrap(),
    );

    let dyn_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0], 
            dimensions: [game.swapchain.dimensions()[0] as f32, game.swapchain.dimensions()[1] as f32],
            depth_range: 0.0..1.0,
        }]),
        ..Default::default()
    };

    let vert_buf = {
        let verts = [
            Vertex::new( 0.0,  0.0),
            Vertex::new(-0.5, -0.5),
            Vertex::new( 0.5, -0.5),

            Vertex::new( 0.0, 0.0),
            Vertex::new(-0.5, 0.5),
            Vertex::new( 0.5, 0.5)
        ];

        CpuAccessibleBuffer::from_iter(game.gpu.clone(), BufferUsage::all(), false, verts.iter().cloned()).unwrap()
    };

    /*
    let layout = pipeline.descriptor_set_layout(0).unwrap();
    let mut pool = PersistentDescriptorSet::start(*layout).build().unwrap();
    let uniform_buf = CpuAccessibleBuffer::from_data(game.gpu.clone(), BufferUsage::all(), false, 5.0 as f32).unwrap();
    let descriptor_set = Arc::new(pool.next().add_buffer(uniform_buf).unwrap().build().unwrap());
    */

    let (time_buf, space_buf, desc) = {
        let layout = pipeline.descriptor_set_layout(0).unwrap();

        let time_buf = CpuAccessibleBuffer::from_data(game.gpu.clone(), BufferUsage::all(), false, 0.0f32).unwrap();

        let space_buf = CpuAccessibleBuffer::from_data(game.gpu.clone(), BufferUsage::all(), false, (0.0f32, 0.0f32)).unwrap();

        let desc = PersistentDescriptorSet::start(layout.clone())
            .add_buffer(time_buf.clone()).unwrap()
            .add_buffer(space_buf.clone()).unwrap()
            .build().unwrap();

        (time_buf, space_buf, Arc::new(desc))
    };

    'running: loop {
        prev_frame_end.as_mut().unwrap().cleanup_finished();

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::MouseMotion { .. } => (),
                Event::MouseButtonDown { x, y, .. } => {
                    let dims = game.swapchain.dimensions();
                    let (w, h) = (dims[0], dims[1]);
                    let ecks = (2*x - w as i32) as f32 / w as f32;
                    let why  = (2*y - h as i32) as f32 / h as f32;
                    *space_buf.write().unwrap() = (ecks, why);
                },
                _ => println!("{:?}", event),
            }
        }

        *time_buf.write().unwrap() += 0.1;

        let (image_num, suboptimal, acquire_future) =
            swapchain::acquire_next_image(game.swapchain.clone(), None).unwrap();

        if suboptimal {
            panic!("suboptimal");
        }

        let fb = game.framebuffers[image_num].clone();

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(game.gpu.clone(), game.queue.family())
            .unwrap()
            .begin_render_pass(fb, false, clear_values.to_vec())
            .unwrap()
            .draw(pipeline.clone(), &dyn_state, vert_buf.clone(), desc.clone(), ())
            .expect("draw call failed")
            .end_render_pass()
            .unwrap()
            .build()
            .unwrap();

        let future = prev_frame_end.take().unwrap()
            .join(acquire_future)
            .then_execute(game.queue.clone(), command_buffer).unwrap()
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
