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
    kernels::{Chunkable, KernelBuff},
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

//This is a simple a (+,-,/) b = c vec op
const BINARY_NO_BROAD_SOURCE: &str = include_str!("binary_no_broad.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct BinaryNoBroadPipelineSpec {
    local_x: u32,
    op: super::BinaryOp,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes)]
#[repr(C, packed)]
pub struct BinarayPushConst {
    start: u32,
    stop: u32,
}

impl PushConstSpec for BinarayPushConst {
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

impl PipelineSpec for BinaryNoBroadPipelineSpec {
    type PushConst = BinarayPushConst;

    fn get_shader(&self) -> CompilationArtifact {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new("binary_no_broad.comp", BINARY_NO_BROAD_SOURCE);
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
        let desc_types: Vec<vk::DescriptorType> = vec![vk::DescriptorType::STORAGE_BUFFER; 3];
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

pub fn run(
    inst: &VultenInstance,
    d_type: VultenDataType,
    op: super::BinaryOp,
    x: &KernelBuff,
    y: &KernelBuff,
    output: &KernelBuff,
    total_elements: i64,
) -> Result<(), &'static str> {
    let spec = BinaryNoBroadPipelineSpec {
        local_x: inst.device_props.sub_group_size.max(1),
        op,
        d_type,
    };
    let pipeline = inst.get_pipeline_from_spec(PipelineSpecs::BinaryNoBroad(spec.clone()));

    let descriptors = inst
        .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
        .unwrap();

    let x_desc_buff = x.get_descriptor_info()?;
    let y_desc_buff = y.get_descriptor_info()?;
    let output_desc_buff = output.get_descriptor_info()?;

    let write_sets = [
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&x_desc_buff),
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&y_desc_buff),
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&output_desc_buff),
    ];
    inst.update_descriptor_sets(&write_sets, &[]);

    let q = inst.get_queue(QueueFlags::COMPUTE);
    let cmd_buffs = inst.create_cmd_buffers(1, &q).unwrap();

    let mut push = BinarayPushConst {
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

    let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
    let fence = inst.create_fence().unwrap();

    inst.submit_queue(&q, &[sub_info], fence).unwrap();
    inst.wait_for_fences(&[fence], true).unwrap();

    inst.destroy_fence(fence);
    inst.free_cmd_buffers(&q, cmd_buffs);
    Ok(())
}
