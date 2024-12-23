use ash::vk::{
    self, AccessFlags, DependencyFlags, DescriptorType, MemoryBarrier, PipelineBindPoint,
    PipelineStageFlags, PushConstantRange, QueueFlags, ShaderStageFlags, SpecializationInfo,
    SpecializationMapEntry, SubmitInfo, WriteDescriptorSet,
};
use shaderc::CompilationArtifact;
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    kernels::Chunkable,
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

use super::KernelInput;

const SSXENT_SOURCE: &str = include_str!("ssxent.comp");
const OP_LOSS: u32 = 0;
const OP_GRAD: u32 = 1;

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct SsxentPipelineSpec {
    local_x: u32,
    d_type: VultenDataType,
    label_d_type: VultenDataType,
}

#[derive(Debug, AsBytes)]
#[repr(C, packed)]
pub struct SsxentPushConst {
    pub start: u32,
    pub stop: u32,
    pub num_logits: u32,
    pub op: u32,
}

impl PushConstSpec for SsxentPushConst {
    fn get_ranges() -> &'static [PushConstantRange] {
        &[PushConstantRange {
            offset: 0,
            stage_flags: ShaderStageFlags::COMPUTE,
            size: std::mem::size_of::<Self>() as u32,
        }]
    }

    #[inline]
    fn get_slice(&self) -> &[u8] {
        let slice: &[u8; 16] = zerocopy::transmute_ref!(self);

        slice
    }
}

impl PipelineSpec for SsxentPipelineSpec {
    type PushConst = SsxentPushConst;

    fn get_shader(&self) -> CompilationArtifact {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new("ssxent.comp", SSXENT_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();
        compiler.add_type_spec(1, self.label_d_type).unwrap();

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
        let desc_types: Vec<vk::DescriptorType> = vec![vk::DescriptorType::STORAGE_BUFFER; 4];
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
    label_d_type: VultenDataType,
    scratch: &KernelInput,
    backprop: &KernelInput,
    labels: &KernelInput,
    loss_fat: &KernelInput,
    grad: &KernelInput,
) -> Result<(), &'static str> {
    let spec = SsxentPipelineSpec {
        local_x: inst.device_props.sub_group_size.max(1),
        d_type,
        label_d_type,
    };
    let pipeline = inst.get_pipeline_from_spec(PipelineSpecs::Ssxent(spec.clone()));

    let descriptors_loss = inst
        .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
        .unwrap();
    let descriptors_grad = inst
        .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
        .unwrap();

    let scratch_desc_buff = scratch.buff.get_descriptor_info()?;
    let backprop_desc_buff = backprop.buff.get_descriptor_info()?;
    let labels_desc_buff = labels.buff.get_descriptor_info()?;
    let loss_fat_desc_buff = loss_fat.buff.get_descriptor_info()?;
    let grad_desc_buff = grad.buff.get_descriptor_info()?;

    let write_sets = [
        //loss
        WriteDescriptorSet::default()
            .dst_set(descriptors_loss.descriptor[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&scratch_desc_buff),
        WriteDescriptorSet::default()
            .dst_set(descriptors_loss.descriptor[0])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&backprop_desc_buff),
        WriteDescriptorSet::default()
            .dst_set(descriptors_loss.descriptor[0])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&labels_desc_buff),
        WriteDescriptorSet::default()
            .dst_set(descriptors_loss.descriptor[0])
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&loss_fat_desc_buff),
        //Grad
        WriteDescriptorSet::default()
            .dst_set(descriptors_grad.descriptor[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&scratch_desc_buff),
        WriteDescriptorSet::default()
            .dst_set(descriptors_grad.descriptor[0])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&backprop_desc_buff),
        WriteDescriptorSet::default()
            .dst_set(descriptors_grad.descriptor[0])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&labels_desc_buff),
        WriteDescriptorSet::default()
            .dst_set(descriptors_grad.descriptor[0])
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&grad_desc_buff),
    ];
    inst.update_descriptor_sets(&write_sets, &[]);

    let q = inst.get_queue(QueueFlags::COMPUTE);
    let cmd_buffs = inst.create_cmd_buffers(1, &q).unwrap();

    let total_elements: i64 = backprop.dims.iter().product();
    let mut push = SsxentPushConst {
        start: 0,
        stop: total_elements as u32,
        num_logits: backprop.dims[1] as u32,
        op: OP_LOSS,
    };

    let mut builder = CommandBufferBuilder::new(cmd_buffs[0], &inst.device)
        .begin()
        .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::COMPUTE,
            pipeline.pipeline_layout,
            0,
            &descriptors_loss.descriptor,
            &[],
        );

    let barrier = MemoryBarrier::default()
        .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

    let chunk_size = inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
    if total_elements > chunk_size {
        let chunks = (0..total_elements).as_chunks(chunk_size);

        for chunk in &chunks {
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

        builder = builder.pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );

        push.op = OP_GRAD;
        builder = builder.bind_descriptor_sets(
            PipelineBindPoint::COMPUTE,
            pipeline.pipeline_layout,
            0,
            &descriptors_grad.descriptor,
            &[],
        );
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

        //loss
        builder = builder
            .push_constants(
                pipeline.pipeline_layout,
                ShaderStageFlags::COMPUTE,
                0,
                push.get_slice(),
            )
            .dispatch(threads, 1, 1);

        builder = builder.pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );

        //grad
        builder = builder.bind_descriptor_sets(
            PipelineBindPoint::COMPUTE,
            pipeline.pipeline_layout,
            0,
            &descriptors_grad.descriptor,
            &[],
        );

        push.op = OP_GRAD;

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
