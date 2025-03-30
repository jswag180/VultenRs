use ash::vk::{
    self, DescriptorType, PipelineBindPoint, PushConstantRange, QueueFlags, ShaderStageFlags,
    SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet,
};
use shaderc::CompilationArtifact;
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    descriptor::VultenDescriptor,
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

use super::{Chunkable, KernelBuff};

const ADD_SUB_SOURCE: &str = include_str!("assign_add_sub.comp");

#[derive(Debug, Clone, Copy)]
pub enum AssignOp {
    Add,
    Sub,
}

impl TryFrom<i32> for AssignOp {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Add),
            1 => Ok(Self::Sub),
            _ => Err(()),
        }
    }
}

impl From<AssignOp> for u32 {
    fn from(value: AssignOp) -> Self {
        match value {
            AssignOp::Add => 0,
            AssignOp::Sub => 1,
        }
    }
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct AssignAddSubPipelineSpec {
    local_x: u32,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct AssignAddSubPushConst {
    start: u32,
    stop: u32,
    op: u32,
}

impl PushConstSpec for AssignAddSubPushConst {
    fn get_ranges() -> &'static [PushConstantRange] {
        &[PushConstantRange {
            offset: 0,
            stage_flags: ShaderStageFlags::COMPUTE,
            size: std::mem::size_of::<Self>() as u32,
        }]
    }

    #[inline]
    fn get_slice(&self) -> &[u8] {
        let slice: &[u8; 12] = zerocopy::transmute_ref!(self);

        slice
    }
}

impl PipelineSpec for AssignAddSubPipelineSpec {
    type PushConst = AssignAddSubPushConst;

    fn get_shader(&self) -> CompilationArtifact {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new("assign_add_sub.comp", ADD_SUB_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.compile()
    }

    fn get_spec_info(&self) -> (Box<[SpecializationMapEntry]>, Vec<u8>) {
        //offset needs to be the offset in the spec_buffer vec not the struct
        let spec_entrys = [SpecializationMapEntry {
            constant_id: 0,
            offset: 0,
            size: std::mem::size_of_val(&self.local_x),
        }];

        let mut spec_buffer: Vec<u8> = Vec::new();
        let local_x_slice = self.local_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_x_slice);

        debug_assert!(spec_buffer.len() <= spec_entrys.iter().fold(0, |acc, x| acc + x.size));

        (Box::new(spec_entrys), spec_buffer)
    }

    fn build_pipeline(&self, inst: &VultenInstance) -> Arc<VultenPipeline> {
        let desc_types: Vec<vk::DescriptorType> = vec![vk::DescriptorType::STORAGE_BUFFER; 2];
        let shader = self.get_shader();
        let spec_info = self.get_spec_info();

        let pipe = inst
            .create_compute_pipeline(
                desc_types,
                shader.as_binary(),
                Some(
                    &SpecializationInfo::default()
                        .map_entries(&spec_info.0)
                        .data(&spec_info.1),
                ),
                Self::PushConst::get_ranges(),
            )
            .unwrap();

        Arc::new(pipe)
    }
}

pub struct AssignAddSubKernel<'a> {
    inst: &'a VultenInstance,
    d_type: VultenDataType,
    op: AssignOp,
    input: Option<KernelBuff<'a>>,
    total_elements: i64,
    output: Option<KernelBuff<'a>>,
    spec: Option<AssignAddSubPipelineSpec>,
}

impl<'a> AssignAddSubKernel<'a> {
    pub fn new(inst: &'a VultenInstance, d_type: VultenDataType, op: AssignOp) -> Self {
        AssignAddSubKernel {
            inst,
            d_type,
            op,
            input: Default::default(),
            total_elements: 0,
            output: Default::default(),
            spec: Default::default(),
        }
    }

    pub fn input(
        mut self,
        input: KernelBuff<'a>,
        total_elements: i64,
    ) -> Result<Self, &'static str> {
        self.input = Some(input);
        self.total_elements = total_elements;

        Ok(self)
    }

    pub fn output(mut self, output: KernelBuff<'a>) -> Result<Self, &'static str> {
        self.output = Some(output);

        Ok(self)
    }

    pub fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::AssignAddSub(spec.clone())))
        } else {
            let spec = AssignAddSubPipelineSpec {
                local_x: self.inst.device_props.sub_group_size.max(1),
                d_type: self.d_type,
            };
            let pipeline = self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::AssignAddSub(spec.clone()));
            self.spec = Some(spec);

            Ok(pipeline)
        }
    }

    pub fn get_descriptors(
        &self,
        pipeline: Arc<VultenPipeline>,
    ) -> Result<VultenDescriptor<'a>, &'static str> {
        let descriptors = self
            .inst
            .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline)
            .or(Err("Could not get descriptor set"))?;

        let input_desc_buff = self
            .input
            .as_ref()
            .ok_or("No input operand")?
            .get_descriptor_info()?;
        let output_desc_buff = self
            .output
            .as_ref()
            .ok_or("No output operand")?
            .get_descriptor_info()?;

        let write_sets = [
            WriteDescriptorSet::default()
                .dst_set(descriptors.descriptor[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&input_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors.descriptor[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_desc_buff),
        ];
        self.inst.update_descriptor_sets(&write_sets, &[]);

        Ok(descriptors)
    }

    pub fn record<'b>(
        &self,
        mut builder: CommandBufferBuilder<'b>,
        pipeline: Arc<VultenPipeline>,
        descriptors: &VultenDescriptor,
    ) -> Result<CommandBufferBuilder<'b>, &'static str> {
        let mut push = AssignAddSubPushConst {
            op: self.op.into(),
            ..Default::default()
        };

        builder = builder
            .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &descriptors.descriptor,
                &[],
            );

        let spec = self.spec.as_ref().ok_or("Missing spec")?;
        let chunk_size = self.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
        let chunks = (0..self.total_elements).as_chunks(chunk_size).into_iter();

        for chunk in chunks {
            push.start = chunk.start as u32;
            push.stop = chunk.end as u32;

            let threads = ((chunk.end - chunk.start) as f32 / spec.local_x as f32)
                .ceil()
                .min((push.stop - push.start) as f32 / spec.local_x as f32)
                as u32;
            builder = builder
                .push_constants(
                    pipeline.pipeline_layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    push.get_slice(),
                )
                .dispatch(threads, 1, 1);
        }

        Ok(builder)
    }

    pub fn run(&mut self) -> Result<(), &'static str> {
        let pipeline = self.get_pipeline()?;
        let descriptors = self.get_descriptors(pipeline.clone())?;
        let q = self.inst.get_queue(QueueFlags::COMPUTE);
        let cmd_buffs = self
            .inst
            .create_cmd_buffers(1, &q)
            .or(Err("Could not create command buffers"))?;
        let builder = CommandBufferBuilder::new(cmd_buffs[0], &self.inst.device).begin();

        self.record(builder, pipeline, &descriptors)?
            .end()
            .build()?;

        let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
        let fence = self.inst.create_fence().or(Err("Could not create fence"))?;

        self.inst
            .submit_queue(&q, &[sub_info], fence)
            .or(Err("Could not submit queue"))?;
        self.inst
            .wait_for_fences(&[fence], true)
            .or(Err("Fence timed out"))?;

        self.inst.destroy_fence(fence);
        self.inst.free_cmd_buffers(&q, cmd_buffs);

        Ok(())
    }
}
