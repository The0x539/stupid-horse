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

    // Beware of alignment discrepancies
    #[repr(C)]
    pub(crate) struct Uniforms {
        pub click_pos: (f32, f32),
        pub window_dims: (f32, f32),
        pub time: f32,
    }
}
