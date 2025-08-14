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
    dims::Dims,
    kernels::{transpose::TransposeKernel, Chunkable},
    memory::{VultenBuffer, VultenBufferType},
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

use super::{ReduceKernel, ReduceKernelVersion, ReduceOp};

const REDUCE_SOURCE: &str = include_str!("reduce_trailing.comp");
const MAX_BLOCK_SIZE: u32 = 32;

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct ReduceTrailingPipelineSpec {
    local_x: u32,
    op: ReduceOp,
    block_size: u32,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct ReduceTrailingPushConst {
    pub ammount: u32,
    pub start: u32,
    pub stop: u32,
}

impl PushConstSpec for ReduceTrailingPushConst {
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

impl PipelineSpec for ReduceTrailingPipelineSpec {
    type PushConst = ReduceTrailingPushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler = compiler::ShaderCompiler::new(REDUCE_SOURCE);
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
            SpecializationMapEntry {
                constant_id: 2,
                offset: 8,
                size: std::mem::size_of::<u32>(),
            },
        ];

        let mut spec_buffer: Vec<u8> = Vec::new();
        let local_x_slice = self.local_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_x_slice);
        let op_slice = (self.op.clone() as u32).to_ne_bytes();
        spec_buffer.extend_from_slice(&op_slice);
        let block_size_slice = (self.block_size).to_ne_bytes();
        spec_buffer.extend_from_slice(&block_size_slice);

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
    ammount: u32,
}

pub struct ReduceKernelTrailing<'a> {
    reduce: ReduceKernel<'a>,
    scratch_buff_1: Option<Arc<VultenBuffer<'a>>>,
    scratch_buff_2: Option<Arc<VultenBuffer<'a>>>,
    transpose: Option<TransposeKernel<'a>>,
    rounds: Vec<ReduceRound>,
    remainder: i64,
    spec: Vec<ReduceTrailingPipelineSpec>,
}

impl<'a> ReduceKernelTrailing<'a> {
    pub fn new(reduce: ReduceKernel<'a>) -> Result<Self, &'static str> {
        let input_dims = reduce.input_dims.ok_or("Missing input dims")?;

        let mut reduce_idxs = reduce.reduce_dims.clone();
        reduce_idxs.sort();
        reduce_idxs.reverse();
        let mut need_transpose = false;
        for (idx, i) in (0..input_dims.len())
            .rev()
            .take(reduce_idxs.len())
            .enumerate()
        {
            if i != reduce_idxs[idx] as usize {
                need_transpose = true;
                break;
            }
        }

        let dims = if need_transpose {
            Self::reorder_vector(input_dims, &reduce_idxs)
        } else {
            input_dims.to_vec()
        };

        let reduce_dims = &dims[dims.len() - reduce_idxs.len()..];
        let remain_dims = &dims[..dims.len() - reduce_idxs.len()];
        let remainder = remain_dims.iter().product::<i64>() as u32;

        let mut rounds = Vec::new();
        let mut ammount = reduce_dims.iter().product::<i64>() as u32;
        loop {
            let mut block_size = if ammount > MAX_BLOCK_SIZE {
                MAX_BLOCK_SIZE
            } else {
                ammount
            };
            for i in (2..MAX_BLOCK_SIZE + 1).rev() {
                if ammount % i == 0 {
                    block_size = i;
                    break;
                }
            }

            rounds.push(ReduceRound {
                block_size,
                ammount,
            });

            if ((remainder * ammount) as f32 / block_size as f32).ceil() as u32 == remainder {
                break;
            }
            ammount = (ammount as f32 / block_size as f32).ceil() as u32;
        }

        let input_size =
            input_dims.iter().product::<i64>() as u64 * reduce.d_type.size_of()? as u64;
        let scratch_buff_1 = if rounds.len() > 1 || need_transpose {
            let size = if need_transpose {
                input_size
            } else {
                let result_size = remainder as i64
                    * (reduce_dims.iter().product::<i64>() as f32 / rounds[0].block_size as f32)
                        .ceil() as i64;
                result_size as u64 * reduce.d_type.size_of()? as u64
            };

            Some(Arc::new(reduce.inst.create_buffer(
                VultenBufferType::Device,
                size,
                false,
                false,
            )))
        } else {
            None
        };
        let scratch_buff_2 = if rounds.len() + (need_transpose as usize) > 2 {
            let size = if need_transpose {
                let result_size = remainder as i64
                    * (reduce_dims.iter().product::<i64>() as f32 / rounds[0].block_size as f32)
                        .ceil() as i64;
                result_size as u64 * reduce.d_type.size_of()? as u64
            } else {
                let result_size = remainder as i64
                    * (rounds[1].ammount as f32 / rounds[1].block_size as f32).ceil() as i64;
                result_size as u64 * reduce.d_type.size_of()? as u64
            };

            Some(Arc::new(reduce.inst.create_buffer(
                VultenBufferType::Device,
                size,
                false,
                false,
            )))
        } else {
            None
        };

        let transpose = if need_transpose {
            let mut transpose_vec = Vec::new();
            for i in 0..input_dims.len() {
                transpose_vec.push(i as i64);
            }
            transpose_vec = Self::reorder_vector(&transpose_vec, &reduce_idxs);

            Some(
                TransposeKernel::new(reduce.inst, reduce.d_type)
                    .input(reduce.input.clone().ok_or("Missing input")?, input_dims)?
                    .output(
                        crate::kernels::KernelBuff::Buff(
                            scratch_buff_1
                                .as_ref()
                                .ok_or("Missing scratch_buff_1")?
                                .clone(),
                        ),
                        Dims::Vec(dims.clone()),
                    )?
                    .transpose(Dims::Vec(transpose_vec))?,
            )
        } else {
            None
        };

        Ok(Self {
            reduce,
            scratch_buff_1,
            scratch_buff_2,
            transpose,
            rounds,
            remainder: remainder as i64,
            spec: Default::default(),
        })
    }

    fn reorder_vector<T: Clone>(original: &[T], indices_to_move: &[u32]) -> Vec<T> {
        // Convert the indices to a set for O(1) lookups
        let indices_set: std::collections::HashSet<usize> =
            indices_to_move.iter().map(|x| *x as usize).collect();

        let mut first_part = Vec::new();
        let mut second_part = Vec::new();

        // Iterate over the original vector and split elements
        for (index, element) in original.iter().enumerate() {
            if indices_set.contains(&index) {
                second_part.push(element.clone());
            } else {
                first_part.push(element.clone());
            }
        }

        // Concatenate the two parts
        first_part.extend(second_part);
        first_part
    }
}

impl<'a> ReduceKernelVersion<'a> for ReduceKernelTrailing<'a> {
    fn get_pipeline(&mut self) -> Result<Vec<Arc<VultenPipeline>>, &'static str> {
        if self.spec.is_empty() {
            for round in &self.rounds {
                let spec = ReduceTrailingPipelineSpec {
                    local_x: self.reduce.inst.device_props.sub_group_size.max(1),
                    op: self.reduce.op.clone(),
                    block_size: round.block_size,
                    d_type: self.reduce.d_type,
                };
                self.spec.push(spec.clone());
            }
        }

        let pipelines: Vec<Arc<VultenPipeline>> = self
            .rounds
            .iter()
            .scan(Vec::new(), |state, x| {
                if !state.contains(&x.block_size) {
                    return Some(x.block_size);
                }

                None
            })
            .map(|block_size| {
                let spec = self
                    .spec
                    .iter()
                    .find(|x| x.block_size == block_size)
                    .unwrap()
                    .clone();

                self.reduce
                    .inst
                    .get_pipeline_from_spec(PipelineSpecs::ReduceTrailing(spec))
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
                if self.transpose.is_some() {
                    write_sets.push(
                        WriteDescriptorSet::default()
                            .dst_set(descriptors[idx].descriptor[0])
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(DescriptorType::STORAGE_BUFFER)
                            .buffer_info(scratch_buff_1_desc_buff.as_ref().unwrap()),
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
                                .buffer_info(scratch_buff_2_desc_buff.as_ref().unwrap()),
                        );
                    }
                } else {
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
                }
            } else if idx == self.rounds.len() - 1 {
                let scratch_idx = (idx + 1 + self.transpose.is_some() as usize) % 2;
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
                let scratch_idx = (idx + 1 + self.transpose.is_some() as usize) % 2;
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

        if let Some(transpose) = self.transpose.as_mut() {
            let transpose_pipeline = transpose.get_pipeline()?;
            let transpose_descriptors = transpose.get_descriptors(transpose_pipeline)?;
            descriptors.push(transpose_descriptors);
        }

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
        if let Some(transpose) = self.transpose.as_mut() {
            let transpose_pipeline = transpose.get_pipeline()?;
            builder = transpose.record(builder, transpose_pipeline, descriptors.last().unwrap())?;
            builder = builder.pipeline_barrier(
                PipelineStageFlags::COMPUTE_SHADER,
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        for (idx, round) in self.rounds.iter().enumerate() {
            let mut push = ReduceTrailingPushConst {
                ammount: round.ammount,
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
            let result_size =
                (round.ammount as f32 / round.block_size as f32).ceil() as i64 * self.remainder;
            let chunks = (0..result_size).as_chunks(chunk_size).into_iter();

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
