use vulkano::format::ClearValue;
use vulkano::format::Format;
use vulkano::framebuffer::RenderPassDesc;
use vulkano::framebuffer::RenderPassDescClearValues;
use vulkano::framebuffer::AttachmentDescription;
use vulkano::framebuffer::PassDescription;
use vulkano::framebuffer::PassDependencyDescription;
use vulkano::image::ImageLayout;
use vulkano::sync::AccessFlagBits;
use vulkano::sync::PipelineStages;

pub struct CustomRenderPassDesc {
    pub color: (Format, u32),
}

unsafe impl RenderPassDesc for CustomRenderPassDesc {
    #[inline]
    fn num_attachments(&self) -> usize { 1 }

    #[inline]
    fn attachment_desc(&self, id: usize) -> Option<AttachmentDescription> {
        match id {
            0 => {
                Some(vulkano::framebuffer::AttachmentDescription {
                    format: self.color.0,
                    samples: self.color.1,
                    load: vulkano::framebuffer::LoadOp::Clear,
                    store: vulkano::framebuffer::StoreOp::Store,
                    stencil_load: vulkano::framebuffer::LoadOp::Clear,
                    stencil_store: vulkano::framebuffer::StoreOp::Store,
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
                let mut desc = PassDescription {
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
    fn dependency_desc(&self, id: usize) -> Option<PassDependencyDescription> {
        None
    }
}

unsafe impl RenderPassDescClearValues<Vec<ClearValue>> for CustomRenderPassDesc {
    fn convert_clear_values(&self, values: Vec<ClearValue>) -> Box<dyn Iterator<Item = ClearValue>> {
        // FIXME: safety checks
        Box::new(values.into_iter())
    }
}
