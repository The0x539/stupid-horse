#![allow(unused_variables, unused_macros, unused_imports, unused_mut)]

use std::sync::Arc;
use std::ffi::CString;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState},
    device::{Device, DeviceExtensions, Features, Queue},
    format::{ClearValue, Format},
    framebuffer::{Framebuffer, FramebufferAbstract, Subpass},
    image::{Dimensions, StorageImage},
    instance::{Instance, RawInstanceExtensions, PhysicalDevice},
    pipeline::{viewport::Viewport, GraphicsPipeline},
    swapchain::{
        self, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform, Swapchain,
    },
    sync,
    sync::GpuFuture,
    VulkanObject,
    descriptor::{
        descriptor_set::PersistentDescriptorSet,
        pipeline_layout::PipelineLayoutAbstract,
    }
};

use image::{ImageBuffer, Rgba};

use sdl2::{Sdl, event::Event, keyboard::Keycode, pixels::Color, video::Window};

mod vs {
    vulkano_shaders::shader! { ty: "vertex", path: "src/shaders/vertex.glsl" }
}
mod fs {
    vulkano_shaders::shader! { ty: "fragment", path: "src/shaders/fragment.glsl" }
}

static DIMS: (u32, u32) = (600, 600);

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

struct Game {
    pub sdl: Sdl,
    pub vulkan: Arc<Instance>,
    pub gpu: Arc<Device>,
    phys_gpu_index: usize,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface<()>>,
    pub window: Window,
}

fn required_extensions(window: &sdl2::video::Window) -> RawInstanceExtensions {
    let ext_names: Vec<&str> = window.vulkan_instance_extensions().unwrap();

    let ext_strs = ext_names.into_iter().map(|s| CString::new(s.as_bytes()).unwrap());

    RawInstanceExtensions::new(ext_strs)
}

impl Game {
    pub fn new() -> Self {
        let sdl = sdl2::init().unwrap();
        let video_subsystem = sdl.video().unwrap();

        let window = video_subsystem
            .window("stupid horse", DIMS.0, DIMS.1)
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

            // One would think this should use window instead of ()
            // but that... breaks thread safety? somehow? why?
            // why does that even matter?
            Arc::new(Surface::from_raw_surface(inst.clone(), raw_surface, ()))
        };

        let gpu_idx = phys_gpu.index();

        Self {
            sdl: sdl,
            vulkan: inst,
            gpu: gpu,
            phys_gpu_index: gpu_idx,
            queue: queue,
            surface: surface,
            window: window,
        }
    }

    pub fn phys_gpu(&self) -> PhysicalDevice {
        PhysicalDevice::from_index(&self.vulkan, self.phys_gpu_index).unwrap()
    }
}

fn main() {
    let game = Game::new();

    let caps = game
        .surface
        .capabilities(game.phys_gpu())
        .expect("failed to get surface capabilities");

    let dims = caps.current_extent.unwrap_or([DIMS.0, DIMS.1]);
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0];

    let (swapchain, images) = Swapchain::new(
        game.gpu.clone(),
        game.surface.clone(),
        caps.min_image_count,
        format.0,
        dims,
        1,
        caps.supported_usage_flags,
        &game.queue,
        SurfaceTransform::Identity,
        alpha,
        PresentMode::Fifo,
        FullscreenExclusive::Default,
        true,
        format.1,
    ).expect("failed to create swapchain");

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            game.gpu.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {color: [color], depth_stencil: {}}
        ).unwrap()
    );

    let framebuffers = images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();

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
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(game.gpu.clone())
            .unwrap(),
    );

    let mut dyn_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0], 
            dimensions: [dims[0] as f32, dims[1] as f32],
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
                    let ecks = (2*x - dims[0] as i32) as f32 / dims[0] as f32;
                    let why  = (2*y - dims[1] as i32) as f32 / dims[1] as f32;
                    *space_buf.write().unwrap() = (ecks, why);
                },
                _ => println!("{:?}", event),
            }
        }

        *time_buf.write().unwrap() += 0.1;

        let (image_num, suboptimal, acquire_future) =
            swapchain::acquire_next_image(swapchain.clone(), None).unwrap();

        let fb = framebuffers[image_num].clone();

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
            .then_swapchain_present(game.queue.clone(), swapchain.clone(), image_num)
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
