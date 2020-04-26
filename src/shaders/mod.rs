use std::sync::Arc;
use vulkano::{
    descriptor::pipeline_layout::PipelineLayoutDesc,
    device::Device,
    pipeline::shader::{GraphicsEntryPoint, ShaderInterfaceDef},
    OomError,
};

pub(crate) trait ShaderAbstract: Sized {
    type I: ShaderInterfaceDef;
    type O: ShaderInterfaceDef;
    type L: PipelineLayoutDesc + Clone + Send + Sync + 'static;
    fn load(device: Arc<Device>) -> Result<Self, OomError>;
    fn main_entry_point(&self) -> GraphicsEntryPoint<(), Self::I, Self::O, Self::L>;
}

macro_rules! shader {
    ($($field:ident: $val:expr),*$(,)?) => {
        vulkano_shaders::shader! {
            $($field: $val),*
        }
        impl crate::shaders::ShaderAbstract for Shader {
            type I = MainInput;
            type O = MainOutput;
            type L = Layout;
            fn load(device: Arc<Device>) -> Result<Self, vulkano::OomError> {
                Self::load(device)
            }
            fn main_entry_point(&self) -> vulkano::pipeline::shader::GraphicsEntryPoint<(), Self::I, Self::O, Self::L> {
                self.main_entry_point()
            }
        }
    };
}

pub(crate) mod bg {
    pub(crate) mod vert {
        shader! {
            ty: "vertex",
            path: "src/shaders/bg.vert"
        }
    }

    pub(crate) mod frag {
        shader! {
            ty: "fragment",
            path: "src/shaders/bg.frag"
        }
    }
}

pub(crate) mod fg {
    pub(crate) mod vert {
        shader! {
            ty: "vertex",
            path: "src/shaders/fg.vert"
        }
    }

    pub(crate) mod frag {
        shader! {
            ty: "fragment",
            path: "src/shaders/fg.frag"
        }
    }

    #[repr(C)]
    pub(crate) struct Uniforms(pub (f32, f32), pub f32, pub f32);
}
