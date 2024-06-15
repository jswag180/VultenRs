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
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    va::VaAddress,
    VultenDataType, VultenInstance,
};

use super::Chunkable;

const UNARY_SOURCE: &str = include_str!("unary.comp");

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum UnaryOp {
    Sqrt,
    Exp,
    Log,
    Square,
    Neg,
    Reciprocal,
    Log1p,
}

const OP_SQRT: u32 = 0;
const OP_EXP: u32 = 1;
const OP_LOG: u32 = 2;
const OP_SQUARE: u32 = 3;
const OP_NEG: u32 = 4;
const OP_RECIPROCAL: u32 = 5;
const OP_LOG1P: u32 = 6;

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
            _ => Err(()),
        }
    }
}

impl From<UnaryOp> for u32 {
    fn from(value: UnaryOp) -> Self {
        match value {
            UnaryOp::Sqrt => OP_SQRT,
            UnaryOp::Exp => OP_EXP,
            UnaryOp::Log => OP_LOG,
            UnaryOp::Square => OP_SQUARE,
            UnaryOp::Neg => OP_NEG,
            UnaryOp::Reciprocal => OP_RECIPROCAL,
            UnaryOp::Log1p => OP_LOG1P,
        }
    }
}

impl UnaryOp {
    pub const fn into_u32(self) -> u32 {
        match self {
            Self::Sqrt => OP_SQRT,
            Self::Exp => OP_EXP,
            Self::Log => OP_LOG,
            Self::Square => OP_SQUARE,
            Self::Neg => OP_NEG,
            Self::Reciprocal => OP_RECIPROCAL,
            Self::Log1p => OP_LOG1P,
        }
    }
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct UnaryPipelineSpec {
    local_x: u32,
    op: UnaryOp,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes)]
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

    fn get_shader(&self) -> CompilationArtifact {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new("unary.comp", UNARY_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.compile()
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
        let op_as_u32: u32 = self.op.clone().into();
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
                shader.as_binary(),
                Some(
                    &SpecializationInfo::builder()
                        .map_entries(&spec_info.0)
                        .data(&spec_info.1)
                        .build(),
                ),
                Self::PushConst::get_ranges(),
            )
            .unwrap();

        Arc::new(pipe)
    }
}

pub fn run(
    inst: &VultenInstance,
    d_type: VultenDataType,
    op: UnaryOp,
    input: VaAddress,
    output: VaAddress,
    total_elements: i64,
) -> Result<(), &'static str> {
    let spec = UnaryPipelineSpec {
        local_x: inst.device_props.sub_group_size.max(1),
        op,
        d_type,
    };
    let pipeline = inst.get_pipeline_from_spec(PipelineSpecs::Unary(spec.clone()));

    let descriptors = inst
        .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
        .unwrap();

    let input_desc_buff = VultenInstance::get_descriptor_info_va(input).unwrap();
    let output_desc_buff = VultenInstance::get_descriptor_info_va(output).unwrap();

    let write_sets = [
        WriteDescriptorSet::builder()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&input_desc_buff.0)
            .build(),
        WriteDescriptorSet::builder()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&output_desc_buff.0)
            .build(),
    ];
    inst.update_descriptor_sets(&write_sets, &[]);

    let q = inst.get_queue(QueueFlags::COMPUTE);
    let cmd_buffs = inst.create_cmd_buffers(1, &q).unwrap();

    let mut push = UnaryPushConst {
        start: 0,
        stop: total_elements as u32,
    };

    let mut builder = CommandBufferBuilder::new(cmd_buffs[0], &inst.device)
        .begin()
        .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::COMPUTE,
            pipeline.pipeline_layout,
            0,
            &descriptors.descriptor,
            &[],
        );

    let chunk_size = inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
    if total_elements > chunk_size {
        let chunks = (0..total_elements).as_chunks(chunk_size).into_iter();

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
    } else {
        let threads = (total_elements as f32 / spec.local_x as f32).ceil() as u32;
        builder = builder
            .push_constants(
                pipeline.pipeline_layout,
                ShaderStageFlags::COMPUTE,
                0,
                push.get_slice(),
            )
            .dispatch(threads, 1, 1);
    }

    builder.end().build().unwrap();

    let sub_info = SubmitInfo::builder().command_buffers(&cmd_buffs).build();
    let fence = inst.create_fence().unwrap();

    inst.submit_queue(&q, &[sub_info], fence).unwrap();
    inst.wait_for_fences(&[fence], true).unwrap();

    inst.destroy_fence(fence);
    inst.free_cmd_buffers(&q, cmd_buffs);
    Ok(())
}
