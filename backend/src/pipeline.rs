use ash::vk::{self, PushConstantRange, SpecializationMapEntry};
use std::sync::Arc;

use crate::{
    kernels::{
        assign_add_sub_variable::AssignAddSubPipelineSpec,
        binary::{
            binary_broad::BinaryBroadPipelineSpec, binary_no_broad::BinaryNoBroadPipelineSpec,
            binary_simple::BinarySimplePipelineSpec,
        },
        conv2d::{
            col2im::Col2ImPipelineSpec, conv2d_gemm::Conv2DGemmPipelineSpec,
            im2col::Im2ColPipelineSpec,
        },
        matmul::{mat_transpose::MatTransposePipelineSpec, MatmulPipelineSpec},
        reduce::reduce_slow::ReducePipelineSpec,
        ssxent::SsxentPipelineSpec,
        transpose::TransposePipelineSpec,
        unary::UnaryPipelineSpec,
    },
    VultenInstance,
};

#[derive(Debug)]
pub enum PiplineCreateError {
    FailedToBuildShaderMod,
    FailedToCreateDescLayout,
    FailedToCreatePipeLayout,
    FailedToCreatePipeline,
}

pub trait PushConstSpec {
    fn get_ranges() -> &'static [PushConstantRange];
    fn get_slice(&self) -> &[u8];
}

pub trait PipelineSpec {
    type PushConst: PushConstSpec;

    /// Get or create pipeline
    fn build_pipeline(&self, inst: &VultenInstance) -> Arc<VultenPipeline>;
    fn get_shader(&self) -> Vec<u32>;
    fn get_spec_info(&self) -> (Box<[SpecializationMapEntry]>, Vec<u8>);
}

#[derive(Eq, Hash, PartialEq, Clone)]
pub enum PipelineSpecs {
    AssignAddSub(AssignAddSubPipelineSpec),
    BinaryNoBroad(BinaryNoBroadPipelineSpec),
    BinarySimple(BinarySimplePipelineSpec),
    BinaryBroad(Box<BinaryBroadPipelineSpec>),
    Unary(UnaryPipelineSpec),
    Matmul(MatmulPipelineSpec),
    MatTranspose(MatTransposePipelineSpec),
    Reduce(ReducePipelineSpec),
    Ssxent(SsxentPipelineSpec),
    Im2Col(Im2ColPipelineSpec),
    Col2Im(Col2ImPipelineSpec),
    Conv2DGemm(Conv2DGemmPipelineSpec),
    Transpose(TransposePipelineSpec),
}

pub struct VultenPipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub shader_mod: vk::ShaderModule,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl super::VultenInstance {
    fn create_shader_mod(
        &self,
        shader_source: &[u32],
    ) -> Result<vk::ShaderModule, PiplineCreateError> {
        let shader_mod_info = vk::ShaderModuleCreateInfo::default().code(shader_source);
        match unsafe { self.device.create_shader_module(&shader_mod_info, None) } {
            Ok(i) => Ok(i),
            Err(_) => Err(PiplineCreateError::FailedToBuildShaderMod),
        }
    }

    fn create_descriptor_set_layout(
        &self,
        buffer_types: Vec<vk::DescriptorType>,
    ) -> Result<vk::DescriptorSetLayout, PiplineCreateError> {
        let mut descriptor_set_layout_bindings: Vec<vk::DescriptorSetLayoutBinding> =
            Vec::with_capacity(buffer_types.len());
        for (i, buff_type) in buffer_types.into_iter().enumerate() {
            descriptor_set_layout_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(buff_type)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .descriptor_count(1),
            );
        }

        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(descriptor_set_layout_bindings.as_slice());
        match unsafe {
            self.device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
        } {
            Ok(i) => Ok(i),
            Err(_) => Err(PiplineCreateError::FailedToCreateDescLayout),
        }
    }

    pub fn create_compute_pipeline(
        &self,
        buffer_types: Vec<vk::DescriptorType>,
        shader_source: &[u32],
        spec_info: Option<&vk::SpecializationInfo>,
        push_const_ranges: &[vk::PushConstantRange],
    ) -> Result<VultenPipeline, PiplineCreateError> {
        let descriptor_set_layout = self.create_descriptor_set_layout(buffer_types)?;

        let descriptor_set_layouts = [descriptor_set_layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(push_const_ranges);
        let pipeline_layout = match unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
        } {
            Ok(i) => i,
            Err(_) => return Err(PiplineCreateError::FailedToCreatePipeLayout),
        };

        let shader_mod = self.create_shader_mod(shader_source)?;

        let mut pipeline_shader_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_mod)
            .name(c"main");
        if let Some(i) = spec_info {
            pipeline_shader_info.p_specialization_info = i
        };

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(pipeline_shader_info)
            .layout(pipeline_layout);
        let pipeline = match unsafe {
            self.device
                .create_compute_pipelines(*self.pipeline_cache, &[pipeline_info], None)
        } {
            Ok(i) => *i.first().unwrap(),
            Err(_) => return Err(PiplineCreateError::FailedToCreatePipeline),
        };

        Ok(VultenPipeline {
            pipeline,
            shader_mod,
            pipeline_layout,
            descriptor_set_layout,
        })
    }

    pub fn free_pipline(&self, pipeline: VultenPipeline) {
        unsafe {
            self.device.destroy_shader_module(pipeline.shader_mod, None);
            self.device
                .destroy_pipeline_layout(pipeline.pipeline_layout, None);
            self.device.destroy_pipeline(pipeline.pipeline, None);
            self.device
                .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
        }
    }

    pub fn get_pipeline_from_spec(&self, spec: PipelineSpecs) -> Arc<VultenPipeline> {
        let locked_map = &mut self.pipelines.upgradable_read();
        if locked_map.contains_key(&spec) {
            locked_map.get(&spec).unwrap().clone()
        } else {
            locked_map.with_upgraded(|m| {
                match spec.clone() {
                    PipelineSpecs::AssignAddSub(pip) => {
                        m.insert(spec.clone(), pip.build_pipeline(self))
                    }
                    PipelineSpecs::BinaryNoBroad(pip) => {
                        m.insert(spec.clone(), pip.build_pipeline(self))
                    }
                    PipelineSpecs::BinarySimple(pip) => {
                        m.insert(spec.clone(), pip.build_pipeline(self))
                    }
                    PipelineSpecs::BinaryBroad(pip) => {
                        m.insert(spec.clone(), pip.build_pipeline(self))
                    }
                    PipelineSpecs::Unary(pip) => m.insert(spec.clone(), pip.build_pipeline(self)),
                    PipelineSpecs::Matmul(pip) => m.insert(spec.clone(), pip.build_pipeline(self)),
                    PipelineSpecs::MatTranspose(pip) => {
                        m.insert(spec.clone(), pip.build_pipeline(self))
                    }
                    PipelineSpecs::Reduce(pip) => m.insert(spec.clone(), pip.build_pipeline(self)),
                    PipelineSpecs::Ssxent(pip) => m.insert(spec.clone(), pip.build_pipeline(self)),
                    PipelineSpecs::Im2Col(pip) => m.insert(spec.clone(), pip.build_pipeline(self)),
                    PipelineSpecs::Col2Im(pip) => m.insert(spec.clone(), pip.build_pipeline(self)),
                    PipelineSpecs::Conv2DGemm(pip) => {
                        m.insert(spec.clone(), pip.build_pipeline(self))
                    }
                    PipelineSpecs::Transpose(pip) => {
                        m.insert(spec.clone(), pip.build_pipeline(self))
                    }
                };

                m.get(&spec).unwrap().clone()
            })
        }
    }
}
