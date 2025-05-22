use ash::vk::{
    self, DescriptorType, PipelineBindPoint, PushConstantRange, QueueFlags, ShaderStageFlags,
    SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet,
};
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

const UNARY_SOURCE: &str = include_str!("unary.comp");

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum UnaryOp {
    Sqrt = 0,
    Exp = 1,
    Log = 2,
    Square = 3,
    Neg = 4,
    Reciprocal = 5,
    Log1p = 6,
    Tanh = 7,
    Relu = 8,
    Rsqrt = 9,
}

const OP_SQRT: u32 = 0;
const OP_EXP: u32 = 1;
const OP_LOG: u32 = 2;
const OP_SQUARE: u32 = 3;
const OP_NEG: u32 = 4;
const OP_RECIPROCAL: u32 = 5;
const OP_LOG1P: u32 = 6;
const OP_TANH: u32 = 7;
const OP_RELU: u32 = 8;
const OP_RSQRT: u32 = 9;

impl TryFrom<u32> for UnaryOp {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            OP_SQRT => Ok(Self::Sqrt),
            OP_EXP => Ok(Self::Exp),
            OP_LOG => Ok(Self::Log),
            OP_SQUARE => Ok(Self::Square),
            OP_NEG => Ok(Self::Neg),
            OP_RECIPROCAL => Ok(Self::Reciprocal),
            OP_LOG1P => Ok(Self::Log1p),
            OP_TANH => Ok(Self::Tanh),
            OP_RELU => Ok(Self::Relu),
            OP_RSQRT => Ok(Self::Rsqrt),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct UnaryPipelineSpec {
    local_x: u32,
    op: UnaryOp,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct UnaryPushConst {
    start: u32,
    stop: u32,
}

impl PushConstSpec for UnaryPushConst {
    fn get_ranges() -> &'static [PushConstantRange] {
        &[PushConstantRange {
            offset: 0,
            stage_flags: ShaderStageFlags::COMPUTE,
            size: std::mem::size_of::<Self>() as u32,
        }]
    }

    #[inline]
    fn get_slice(&self) -> &[u8] {
        let slice: &[u8; 8] = zerocopy::transmute_ref!(self);

        slice
    }
}

impl PipelineSpec for UnaryPipelineSpec {
    type PushConst = UnaryPushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler = compiler::ShaderCompiler::new(UNARY_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.compile().unwrap()
    }

    fn get_spec_info(&self) -> (Box<[SpecializationMapEntry]>, Vec<u8>) {
        //offset needs to be the offset in the spec_buffer vec not the struct
        let spec_entrys = [
            SpecializationMapEntry {
                constant_id: 0,
                offset: 0,
                size: std::mem::size_of_val(&self.local_x),
            },
            SpecializationMapEntry {
                constant_id: 1,
                offset: 4,
                size: std::mem::size_of::<u32>(),
            },
        ];

        let mut spec_buffer: Vec<u8> = Vec::new();
        let local_x_slice = self.local_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_x_slice);
        let op_as_u32: u32 = self.op.clone() as u32;
        let op_slice = op_as_u32.to_ne_bytes();
        spec_buffer.extend_from_slice(&op_slice);

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
                &shader,
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

pub struct UnaryKernel<'a> {
    inst: &'a VultenInstance,
    d_type: VultenDataType,
    op: UnaryOp,
    input: Option<KernelBuff<'a>>,
    total_elements: i64,
    output: Option<KernelBuff<'a>>,
    spec: Option<UnaryPipelineSpec>,
}

impl<'a> UnaryKernel<'a> {
    pub fn new(inst: &'a VultenInstance, d_type: VultenDataType, op: UnaryOp) -> Self {
        UnaryKernel {
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
                .get_pipeline_from_spec(PipelineSpecs::Unary(spec.clone())))
        } else {
            let spec = UnaryPipelineSpec {
                local_x: self.inst.device_props.sub_group_size.max(1),
                op: self.op.clone(),
                d_type: self.d_type,
            };
            let pipeline = self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Unary(spec.clone()));
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
        let mut push = UnaryPushConst::default();

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

            let threads = ((chunk.end - chunk.start) as f32 / spec.local_x as f32).ceil() as u32;
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
