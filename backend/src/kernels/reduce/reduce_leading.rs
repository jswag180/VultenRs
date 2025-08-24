use ash::vk::{
    self, AccessFlags, DependencyFlags, DescriptorBufferInfo, DescriptorType, MemoryBarrier,
    PipelineBindPoint, PipelineStageFlags, PushConstantRange, QueueFlags, ShaderStageFlags,
    SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet,
};
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    descriptor::VultenDescriptor,
    kernels::Chunkable,
    memory::{VultenBuffer, VultenBufferType},
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

use super::{ReduceKernel, ReduceKernelVersion, ReduceOp};

const REDUCE_SOURCE: &str = include_str!("reduce_leading.comp");
const MAX_BLOCK_SIZE: u32 = 4;
const MAX_BLOCK_GROUP_SIZE: u32 = 128;

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct ReduceLeadingPipelineSpec {
    local_x: u32,
    op: ReduceOp,
    block_size: u32,
    block_group_size: u32,
    block_groups: u32,
    reduce_size: u32,
    remainder: u32,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct ReduceLeadingPushConst {
    pub start: u32,
    pub stop: u32,
}

impl PushConstSpec for ReduceLeadingPushConst {
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

impl PipelineSpec for ReduceLeadingPipelineSpec {
    type PushConst = ReduceLeadingPushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler = compiler::ShaderCompiler::new(REDUCE_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.add_define("BLOCK_SIZE".to_string(), Some(self.block_size.to_string()));
        compiler.add_define("BG".to_string(), Some(self.block_group_size.to_string()));
        compiler.add_define("BGS".to_string(), Some(self.block_groups.to_string()));
        compiler.add_define(
            "REDUCE_SIZE".to_string(),
            Some(self.reduce_size.to_string()),
        );
        compiler.add_define("REMAINDER".to_string(), Some(self.remainder.to_string()));

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
        let op_slice = (self.op.clone() as u32).to_ne_bytes();
        spec_buffer.extend_from_slice(&op_slice);

        debug_assert!(spec_buffer.len() <= spec_entrys.iter().fold(0, |acc, x| acc + x.size));

        (Box::new(spec_entrys), spec_buffer)
    }

    fn build_pipeline(&self, inst: &VultenInstance) -> Arc<VultenPipeline> {
        let desc_types: Vec<vk::DescriptorType> = vec![
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
        ];
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

#[derive(Debug)]
struct ReduceRound {
    block_size: u32,
    block_group_size: u32,
    block_groups: u32,
    reduce_size: u32,
    remainder: u32,
}

pub struct ReduceKernelLeading<'a> {
    reduce: ReduceKernel<'a>,
    scratch_buff_1: Option<Arc<VultenBuffer<'a>>>,
    scratch_buff_2: Option<Arc<VultenBuffer<'a>>>,
    rounds: Vec<ReduceRound>,
    spec: Vec<ReduceLeadingPipelineSpec>,
}

impl<'a> ReduceKernelLeading<'a> {
    pub fn new(reduce: ReduceKernel<'a>) -> Result<Self, &'static str> {
        let input_dims = reduce.input_dims.ok_or("Missing input dims")?;
        let reduce_idxs = &reduce.reduce_dims;

        let reduce_dims = &input_dims[..reduce_idxs.len()];
        let remain_dims = &input_dims[reduce_idxs.len()..];
        let mut reduce_size = reduce_dims.iter().product::<i64>() as u32;
        let remainder = remain_dims.iter().product::<i64>() as u32;

        let mut rounds = Vec::new();
        let mut block_size = 1;
        for i in (0..MAX_BLOCK_SIZE + 1).rev() {
            if remainder % i == 0 {
                block_size = i;
                break;
            }
        }
        loop {
            let mut block_group_size = 2;
            for i in (2..MAX_BLOCK_GROUP_SIZE + 1).rev() {
                if reduce_size % i == 0 {
                    block_group_size = i;
                    break;
                }
            }

            let block_groups = if reduce_size % block_group_size != 0 {
                (((reduce_size * remainder) / block_size) as f32 / block_group_size as f32).ceil()
                    as u32
            } else {
                ((reduce_size * remainder) / block_size) / block_group_size
            };

            rounds.push(ReduceRound {
                block_size,
                block_group_size,
                block_groups,
                reduce_size,
                remainder,
            });

            reduce_size /= block_group_size;
            if reduce_size == 1 {
                break;
            }
        }

        let scratch_buff_1 = if rounds.len() > 1 {
            let round = &rounds[0];
            let out_ele = round.block_groups * round.block_size;
            let size = out_ele as u64 * reduce.d_type.size_of()? as u64;

            Some(Arc::new(reduce.inst.create_buffer(
                VultenBufferType::Device,
                size,
                false,
                false,
            )))
        } else {
            None
        };

        let scratch_buff_2 = if rounds.len() > 2 {
            let round = &rounds[1];
            let out_ele = round.block_groups * round.block_size;
            let size = out_ele as u64 * reduce.d_type.size_of()? as u64;

            Some(Arc::new(reduce.inst.create_buffer(
                VultenBufferType::Device,
                size,
                false,
                false,
            )))
        } else {
            None
        };

        Ok(ReduceKernelLeading {
            reduce,
            scratch_buff_1,
            scratch_buff_2,
            rounds,
            spec: Default::default(),
        })
    }
}

impl<'a> ReduceKernelVersion<'a> for ReduceKernelLeading<'a> {
    fn get_pipeline(&mut self) -> Result<Vec<Arc<VultenPipeline>>, &'static str> {
        if self.spec.is_empty() {
            for round in &self.rounds {
                let spec = ReduceLeadingPipelineSpec {
                    local_x: self.reduce.inst.device_props.sub_group_size.max(1),
                    op: self.reduce.op.clone(),
                    d_type: self.reduce.d_type,

                    block_size: round.block_size,
                    block_group_size: round.block_group_size,
                    block_groups: round.block_groups,
                    reduce_size: round.reduce_size,
                    remainder: round.remainder,
                };
                self.spec.push(spec.clone());
            }
        }

        let pipelines: Vec<Arc<VultenPipeline>> = self
            .spec
            .iter()
            .map(|spec| {
                self.reduce
                    .inst
                    .get_pipeline_from_spec(PipelineSpecs::ReduceLeading(spec.clone()))
            })
            .collect();

        Ok(pipelines)
    }

    fn get_descriptors(
        &mut self,
        pipeline: &[Arc<VultenPipeline>],
    ) -> Result<Vec<VultenDescriptor<'a>>, &'static str> {
        let mut descriptors = Vec::new();
        for _ in 0..self.rounds.len() {
            let reduce_descriptors = self
                .reduce
                .inst
                .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline[0].clone())
                .or(Err("Could not get descriptor set"))?;
            descriptors.push(reduce_descriptors);
        }

        let input_desc_buff = self
            .reduce
            .input
            .as_ref()
            .ok_or("No input operand")?
            .get_descriptor_info()?;
        let output_desc_buff = self
            .reduce
            .output
            .as_ref()
            .ok_or("No output operand")?
            .get_descriptor_info()?;
        let scratch_buff_1_desc_buff = self.scratch_buff_1.as_ref().map(|scratch| {
            [DescriptorBufferInfo::default()
                .buffer(scratch.vk_buffer)
                .offset(0)
                .range(scratch.size)]
        });
        let scratch_buff_2_desc_buff = self.scratch_buff_2.as_ref().map(|scratch| {
            [DescriptorBufferInfo::default()
                .buffer(scratch.vk_buffer)
                .offset(0)
                .range(scratch.size)]
        });

        let mut write_sets = Vec::new();
        let mut scratches = Vec::new();
        if let Some(scratch) = &scratch_buff_1_desc_buff {
            scratches.push(scratch);
        }
        if let Some(scratch) = &scratch_buff_2_desc_buff {
            scratches.push(scratch);
        }
        for (idx, _) in self.rounds.iter().enumerate() {
            if idx == 0 {
                write_sets.push(
                    WriteDescriptorSet::default()
                        .dst_set(descriptors[idx].descriptor[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&input_desc_buff),
                );
                if self.rounds.len() == 1 {
                    write_sets.push(
                        WriteDescriptorSet::default()
                            .dst_set(descriptors[idx].descriptor[0])
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&output_desc_buff),
                    );
                } else {
                    write_sets.push(
                        WriteDescriptorSet::default()
                            .dst_set(descriptors[idx].descriptor[0])
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(DescriptorType::STORAGE_BUFFER)
                            .buffer_info(scratch_buff_1_desc_buff.as_ref().unwrap()),
                    );
                }
            } else if idx == self.rounds.len() - 1 {
                let scratch_idx = (idx + 1) % 2;
                write_sets.push(
                    WriteDescriptorSet::default()
                        .dst_set(descriptors[idx].descriptor[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .buffer_info(scratches[scratch_idx]),
                );
                write_sets.push(
                    WriteDescriptorSet::default()
                        .dst_set(descriptors[idx].descriptor[0])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&output_desc_buff),
                );
            } else {
                let scratch_idx = (idx + 1) % 2;
                write_sets.push(
                    WriteDescriptorSet::default()
                        .dst_set(descriptors[idx].descriptor[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .buffer_info(scratches[scratch_idx]),
                );
                write_sets.push(
                    WriteDescriptorSet::default()
                        .dst_set(descriptors[idx].descriptor[0])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(DescriptorType::STORAGE_BUFFER)
                        .buffer_info(scratches[scratch_idx ^ 1]),
                );
            }
        }

        self.reduce.inst.update_descriptor_sets(&write_sets, &[]);

        Ok(descriptors)
    }

    fn record<'b>(
        &mut self,
        mut builder: CommandBufferBuilder<'b>,
        pipeline: &[Arc<VultenPipeline>],
        descriptors: &[VultenDescriptor],
    ) -> Result<CommandBufferBuilder<'b>, &'static str> {
        let barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
            .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

        for (idx, _round) in self.rounds.iter().enumerate() {
            let mut push = ReduceLeadingPushConst {
                ..Default::default()
            };

            builder = builder
                .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline[idx].clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::COMPUTE,
                    pipeline[idx].pipeline_layout,
                    0,
                    &descriptors[idx].descriptor,
                    &[],
                );

            let spec = &self.spec[idx];
            let chunk_size =
                self.reduce.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
            let chunks = (0..spec.block_groups as i64)
                .as_chunks(chunk_size)
                .into_iter();

            for chunk in chunks {
                push.start = chunk.start as u32;
                push.stop = chunk.end as u32;

                let threads =
                    ((chunk.end - chunk.start) as f32 / spec.local_x as f32).ceil() as u32;
                builder = builder
                    .push_constants(
                        pipeline[idx].pipeline_layout,
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
            }
        }

        Ok(builder)
    }

    fn run(&mut self) -> Result<(), &'static str> {
        let pipeline = self.get_pipeline()?;
        let descriptors = self.get_descriptors(&pipeline)?;
        let q = self.reduce.inst.get_queue(QueueFlags::COMPUTE);
        let cmd_buffs = self
            .reduce
            .inst
            .create_cmd_buffers(1, &q)
            .or(Err("Could not create command buffers"))?;
        let builder = CommandBufferBuilder::new(cmd_buffs[0], &self.reduce.inst.device).begin();

        self.record(builder, &pipeline, &descriptors)?
            .end()
            .build()?;

        let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
        let fence = self
            .reduce
            .inst
            .create_fence()
            .or(Err("Could not create fence"))?;

        self.reduce
            .inst
            .submit_queue(&q, &[sub_info], fence)
            .or(Err("Could not submit queue"))?;
        self.reduce
            .inst
            .wait_for_fences(&[fence], true)
            .or(Err("Fence timed out"))?;

        self.reduce.inst.destroy_fence(fence);
        self.reduce.inst.free_cmd_buffers(&q, cmd_buffs);

        Ok(())
    }
}
