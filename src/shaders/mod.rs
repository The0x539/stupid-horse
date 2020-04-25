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
