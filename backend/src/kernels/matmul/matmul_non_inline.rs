use ash::vk::{
    AccessFlags, DependencyFlags, DescriptorType, MemoryBarrier, PipelineBindPoint,
    PipelineStageFlags, QueueFlags, ShaderStageFlags, SubmitInfo, WriteDescriptorSet,
};
use std::sync::Arc;

use crate::{
    cmd_buff::CommandBufferBuilder,
    descriptor::VultenDescriptor,
    kernels::{Chunkable, KernelBuff},
    pipeline::{PipelineSpecs, PushConstSpec, VultenPipeline},
};

use super::{
    transpose::TransposeKernel, MatMulKernel, MatMulKernelVersion, MatmulPipelineSpec,
    MatmulPushConst, BROADCAST_A, BROADCAST_B, BROADCAST_NONE,
};

pub struct MatMulKernelNonInline<'a> {
    matmul: MatMulKernel<'a>,
    spec: Option<MatmulPipelineSpec>,
    transpose_a: Option<TransposeKernel<'a>>,
    transpose_b: Option<TransposeKernel<'a>>,
    transpose_a_buff: Option<KernelBuff<'a>>,
    transpose_b_buff: Option<KernelBuff<'a>>,
}

impl<'a> MatMulKernelNonInline<'a> {
    pub fn new(matmul: MatMulKernel<'a>) -> Result<Self, &'static str> {
        let transpose_a_buff;
        let transpose_a = if matmul.a_transpose {
            let mut transpose = TransposeKernel::new(matmul.inst, matmul.d_type);
            let buff = matmul.a.as_ref().ok_or("Missing a")?.clone();
            let buff_info = buff.get_descriptor_info()?;
            transpose_a_buff = Some(KernelBuff::Buff(Arc::new(matmul.inst.create_buffer(
                crate::memory::VultenBufferType::Device,
                buff_info[0].range,
                false,
                false,
            ))));
            transpose
                .input(buff, matmul.a_dims.ok_or("Missing a dims")?)?
                .output(transpose_a_buff.as_ref().ok_or("Missing a buff")?.clone())?;
            Some(transpose)
        } else {
            transpose_a_buff = None;
            None
        };

        let transpose_b_buff;
        let transpose_b = if matmul.b_transpose {
            let mut transpose = TransposeKernel::new(matmul.inst, matmul.d_type);
            let buff = matmul.b.as_ref().ok_or("Missing b")?.clone();
            let buff_info = buff.get_descriptor_info()?;
            transpose_b_buff = Some(KernelBuff::Buff(Arc::new(matmul.inst.create_buffer(
                crate::memory::VultenBufferType::Device,
                buff_info[0].range,
                false,
                false,
            ))));
            transpose
                .input(buff, matmul.b_dims.ok_or("Missing b dims")?)?
                .output(transpose_b_buff.as_ref().ok_or("Missing b buff")?.clone())?;
            Some(transpose)
        } else {
            transpose_b_buff = None;
            None
        };

        Ok(Self {
            matmul,
            spec: Default::default(),
            transpose_a,
            transpose_b,
            transpose_a_buff,
            transpose_b_buff,
        })
    }
}

impl<'a> MatMulKernelVersion<'a> for MatMulKernelNonInline<'a> {
    fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .matmul
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Matmul(spec.clone())))
        } else {
            let a_dims = self.matmul.a_dims.as_ref().ok_or("Missing a dims")?;
            let b_dims = self.matmul.b_dims.as_ref().ok_or("Missing b dims")?;
            let offset = a_dims.len() - 2;
            let mat_a_post: (i64, i64) = if self.matmul.a_transpose {
                (a_dims[1 + offset], a_dims[offset])
            } else {
                (a_dims[offset], a_dims[1 + offset])
            };
            let mat_b_post: (i64, i64) = if self.matmul.b_transpose {
                (b_dims[1 + offset], b_dims[offset])
            } else {
                (b_dims[offset], b_dims[1 + offset])
            };
            let broadcast = if offset == 0 || a_dims[0] == b_dims[0] {
                BROADCAST_NONE
            } else if a_dims[0] == 1 {
                BROADCAST_A
            } else {
                BROADCAST_B
            };

            let spec = MatmulPipelineSpec {
                local_x: self.matmul.inst.device_props.sub_group_size.max(1),
                block_size_x: self.matmul.block_dims.0,
                block_size_y: self.matmul.block_dims.1,
                bk_cont: mat_a_post.1 as u32 / self.matmul.block_dims.0,
                a_x: mat_a_post.0 as u32,
                a_y: mat_a_post.1 as u32,
                b_x: mat_b_post.0 as u32,
                b_y: mat_b_post.1 as u32,
                inline_trans_a: false,
                inline_trans_b: false,
                bk_num_y: self.matmul.num_blocks.1 as u32,
                broadcast,
                d_type: self.matmul.d_type,
            };

            let pipeline = self
                .matmul
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Matmul(spec.clone()));
            self.spec = Some(spec);

            Ok(pipeline)
        }
    }

    fn get_descriptors(
        &mut self,
        pipeline: Arc<VultenPipeline>,
    ) -> Result<Vec<VultenDescriptor<'a>>, &'static str> {
        let mut descriptors = Vec::new();
        let descriptor = self
            .matmul
            .inst
            .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline)
            .or(Err("Could not get descriptor set"))?;
        descriptors.push(descriptor);

        let a_desc_buff = self
            .matmul
            .a
            .as_ref()
            .ok_or("No a operand")?
            .get_descriptor_info()?;
        let b_desc_buff = self
            .matmul
            .b
            .as_ref()
            .ok_or("No b operand")?
            .get_descriptor_info()?;
        let output_desc_buff = self
            .matmul
            .output
            .as_ref()
            .ok_or("No output operand")?
            .get_descriptor_info()?;

        let mut write_sets: Vec<WriteDescriptorSet> = Vec::new();

        let transpose_a_desc_buff;
        if let Some(transpose) = self.transpose_a.as_mut() {
            let pipe = transpose.get_pipeline()?;
            let desc = transpose.get_descriptors(pipe)?;
            transpose_a_desc_buff = self
                .transpose_a_buff
                .as_ref()
                .unwrap()
                .get_descriptor_info()?;
            write_sets.push(
                WriteDescriptorSet::default()
                    .dst_set(descriptors[0].descriptor[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&transpose_a_desc_buff),
            );

            descriptors.push(desc);
        } else {
            write_sets.push(
                WriteDescriptorSet::default()
                    .dst_set(descriptors[0].descriptor[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&a_desc_buff),
            );
        }

        let transpose_b_desc_buff;
        if let Some(transpose) = self.transpose_b.as_mut() {
            let pipe = transpose.get_pipeline()?;
            let desc = transpose.get_descriptors(pipe)?;
            transpose_b_desc_buff = self
                .transpose_b_buff
                .as_ref()
                .unwrap()
                .get_descriptor_info()?;
            write_sets.push(
                WriteDescriptorSet::default()
                    .dst_set(descriptors[0].descriptor[0])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&transpose_b_desc_buff),
            );

            descriptors.push(desc);
        } else {
            write_sets.push(
                WriteDescriptorSet::default()
                    .dst_set(descriptors[0].descriptor[0])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&b_desc_buff),
            );
        }

        write_sets.push(
            WriteDescriptorSet::default()
                .dst_set(descriptors[0].descriptor[0])
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_desc_buff),
        );

        self.matmul.inst.update_descriptor_sets(&write_sets, &[]);

        Ok(descriptors)
    }

    fn record<'b>(
        &mut self,
        mut builder: CommandBufferBuilder<'b>,
        pipeline: Arc<VultenPipeline>,
        descriptors: &[VultenDescriptor],
    ) -> Result<CommandBufferBuilder<'b>, &'static str> {
        let mut descriptors = descriptors.iter().rev();

        if let Some(transpose) = self.transpose_a.as_mut() {
            let transpose_pipeline = transpose.get_pipeline()?;
            builder = transpose.record(builder, transpose_pipeline, descriptors.next().unwrap())?;
        }
        if let Some(transpose) = self.transpose_b.as_mut() {
            let transpose_pipeline = transpose.get_pipeline()?;
            builder = transpose.record(builder, transpose_pipeline, descriptors.next().unwrap())?;
        }

        if self.matmul.a_transpose || self.matmul.b_transpose {
            let transpose_barrier = MemoryBarrier::default()
                .src_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
                .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

            builder = builder.pipeline_barrier(
                PipelineStageFlags::COMPUTE_SHADER,
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(),
                &[transpose_barrier],
                &[],
                &[],
            );
        }

        let mut push = MatmulPushConst::default();
        builder = builder
            .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &descriptors.next().unwrap().descriptor,
                &[],
            );

        let spec = self.spec.as_ref().ok_or("Missing spec")?;
        let num_batch = match spec.broadcast {
            BROADCAST_A => self.matmul.b_dims.as_ref().ok_or("Missing b dims")?[0],
            BROADCAST_B => self.matmul.a_dims.as_ref().ok_or("Missing a dims")?[0],
            BROADCAST_NONE => {
                let dims = self.matmul.a_dims.as_ref().ok_or("Missing a dims")?;
                if dims.len() == 2 {
                    1
                } else {
                    dims[0]
                }
            }
            _ => return Err("Invalid broadcast"),
        };
        let chunk_size: i64 =
            self.matmul.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
        let chunks = (0..self.matmul.num_blocks.0 * self.matmul.num_blocks.1).as_chunks(chunk_size);
        for i in 0..num_batch {
            push.offset = i as u32;
            for chunk in &chunks {
                push.start_x = chunk.start as u32;
                push.stop_x = chunk.end as u32;

                let threads =
                    ((chunk.end - chunk.start) as f32 / spec.local_x as f32).ceil() as u32;
                builder = builder
                    .push_constants(
                        pipeline.pipeline_layout,
                        ShaderStageFlags::COMPUTE,
                        0,
                        push.get_slice(),
                    )
                    .dispatch(threads, 1, 1);
            }
        }

        Ok(builder)
    }

    fn run(&mut self) -> Result<(), &'static str> {
        let pipeline = self.get_pipeline()?;
        let descriptors = self.get_descriptors(pipeline.clone())?;
        let q = self.matmul.inst.get_queue(QueueFlags::COMPUTE);
        let cmd_buffs = self
            .matmul
            .inst
            .create_cmd_buffers(1, &q)
            .or(Err("Could not create command buffers"))?;
        let builder = CommandBufferBuilder::new(cmd_buffs[0], &self.matmul.inst.device).begin();

        self.record(builder, pipeline, &descriptors)?
            .end()
            .build()?;

        let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
        let fence = self
            .matmul
            .inst
            .create_fence()
            .or(Err("Could not create fence"))?;

        self.matmul
            .inst
            .submit_queue(&q, &[sub_info], fence)
            .or(Err("Could not submit queue"))?;
        self.matmul
            .inst
            .wait_for_fences(&[fence], true)
            .or(Err("Fence timed out"))?;

        self.matmul.inst.destroy_fence(fence);
        self.matmul.inst.free_cmd_buffers(&q, cmd_buffs);

        Ok(())
    }
}
