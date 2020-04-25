pub(crate) mod bg {
    pub(crate) mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/bg.vert"
        }
    }

    pub(crate) mod frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/bg.frag"
        }
    }
}

pub(crate) mod fg {
    pub(crate) mod vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/fg.vert"
        }
    }

    pub(crate) mod frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/fg.frag"
        }
    }

    #[repr(C)]
    pub(crate) struct Uniforms {
        pub click_pos: (f32, f32),
        pub time: f32,
        pub scale: f32,
    }
}
