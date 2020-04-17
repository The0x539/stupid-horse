use vulkano::{
    format::{ClearValue, Format},
    framebuffer::{
        RenderPassDesc,
        RenderPassDescClearValues,
        AttachmentDescription,
        PassDescription,
        PassDependencyDescription,
        LoadOp,
        StoreOp,
    },
    image::ImageLayout,
};

pub struct Desc {
    pub color: (Format, u32),
}

unsafe impl RenderPassDesc for Desc {
    #[inline]
    fn num_attachments(&self) -> usize { 1 }

    #[inline]
    fn attachment_desc(&self, id: usize) -> Option<AttachmentDescription> {
        match id {
            0 => {
                Some(AttachmentDescription {
                    format: self.color.0,
                    samples: self.color.1,
                    load: LoadOp::Clear,
                    store: StoreOp::Store,
                    stencil_load: LoadOp::Clear,
                    stencil_store: StoreOp::Store,
                    initial_layout: ImageLayout::ColorAttachmentOptimal,
                    final_layout: ImageLayout::ColorAttachmentOptimal,
                })
            },
            _ => None,
        }
    }

    #[inline]
    fn num_subpasses(&self) -> usize { 1 }

    #[inline]
    fn subpass_desc(&self, id: usize) -> Option<PassDescription> {
        match id {
            0 => {
                let desc = PassDescription {
                    color_attachments: vec![(0, ImageLayout::ColorAttachmentOptimal)],
                    depth_stencil: None,
                    input_attachments: vec![],
                    resolve_attachments: vec![],
                    preserve_attachments: vec![],
                };

                assert!(desc.resolve_attachments.is_empty() ||
                        desc.resolve_attachments.len() == desc.color_attachments.len());
                Some(desc)
            },
            _ => None,
        }
    }

    #[inline]
    fn num_dependencies(&self) -> usize { 0 }

    #[inline]
    fn dependency_desc(&self, _id: usize) -> Option<PassDependencyDescription> {
        None
    }
}

unsafe impl RenderPassDescClearValues<Vec<ClearValue>> for Desc {
    fn convert_clear_values(&self, values: Vec<ClearValue>) -> Box<dyn Iterator<Item = ClearValue>> {
        // FIXME: safety checks
        Box::new(values.into_iter())
    }
}
