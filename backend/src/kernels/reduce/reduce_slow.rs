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
    utills::calculate_strdies,
    VultenDataType, VultenInstance,
};

use super::{ReduceKernel, ReduceKernelVersion, ReduceOp};

const RELU_SOURCE: &str = include_str!("reduce.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct ReducePipelineSpec {
    local_x: u32,
    op: ReduceOp,
    max_reduce_dims: usize,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct ReducePushConst {
    pub offset: u32,
    pub start: u32,
    pub stop: u32,
}

impl PushConstSpec for ReducePushConst {
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

impl PipelineSpec for ReducePipelineSpec {
    type PushConst = ReducePushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler = compiler::ShaderCompiler::new(RELU_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();
        compiler.add_define(
            "MAX_REDUCE_DIMS".into(),
            Some(self.max_reduce_dims.to_string()),
        );

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
            vk::DescriptorType::UNIFORM_BUFFER,
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

pub struct ReduceKernelSlow<'a> {
    reduce: ReduceKernel<'a>,
    stride_uniform: VultenBuffer<'a>,
    scratch_buffs: Vec<VultenBuffer<'a>>,
    spec: Option<ReducePipelineSpec>,
}

impl<'a> ReduceKernelSlow<'a> {
    pub fn new(reduce: ReduceKernel<'a>) -> Result<Self, &'static str> {
        let num_sets = reduce.reduce_dims.len();
        let stride_uniform = reduce.inst.create_buffer(
            VultenBufferType::Uiniform,
            (std::mem::size_of::<u32>() * 4 * num_sets)
                .try_into()
                .unwrap(),
            false,
            true,
        );
        let stride_uniform_ptr = stride_uniform.get_mapped_ptr().unwrap() as *mut u32;
        let stride_uniform_slice =
            unsafe { std::slice::from_raw_parts_mut(stride_uniform_ptr, 4 * num_sets) };
        let mut dims = reduce.input_dims.ok_or("Missing input dims")?.to_vec();
        for (i, axis) in reduce.reduce_dims.iter().enumerate() {
            let index = i * 4;
            let strides = calculate_strdies(&dims);

            stride_uniform_slice[index] = dims[*axis as usize] as u32;
            stride_uniform_slice[index + 1] = strides[*axis as usize] as u32;
            stride_uniform_slice[index + 2] = strides[*axis as usize + 1] as u32;
            stride_uniform_slice[index + 3] = 0;

            dims.remove(*axis as usize);
        }

        Ok(Self {
            reduce,
            stride_uniform,
            scratch_buffs: Default::default(),
            spec: Default::default(),
        })
    }
}

impl<'a> ReduceKernelVersion<'a> for ReduceKernelSlow<'a> {
    fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .reduce
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Reduce(spec.clone())))
        } else {
            let spec = ReducePipelineSpec {
                local_x: self.reduce.inst.device_props.sub_group_size.max(1),
                op: self.reduce.op.clone(),
                max_reduce_dims: self.reduce.reduce_dims.len().max(16),
                d_type: self.reduce.d_type,
            };

            let pipeline = self
                .reduce
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Reduce(spec.clone()));
            self.spec = Some(spec);

            Ok(pipeline)
        }
    }

    fn get_descriptors(
        &mut self,
        pipeline: Arc<VultenPipeline>,
    ) -> Result<Vec<VultenDescriptor<'a>>, &'static str> {
        let input_buff = self
            .reduce
            .input
            .as_ref()
            .ok_or("Missing input")?
            .get_buffer()?;
        let input_dims = self.reduce.input_dims.ok_or("Missing input dims")?;
        let output_buff = self
            .reduce
            .output
            .as_ref()
            .ok_or("Missing input")?
            .get_buffer()?;

        let num_sets = self.reduce.reduce_dims.len();
        let mut descriptor_sets: Vec<VultenDescriptor> = Vec::with_capacity(num_sets);
        for _ in 0..num_sets {
            descriptor_sets.push(
                self.reduce
                    .inst
                    .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
                    .unwrap(),
            );
        }
        let mut scratch_buffs: Vec<VultenBuffer> = Vec::with_capacity(num_sets - 1);
        let mut size_to_shave: u64 = 1;
        for i in self.reduce.reduce_dims.iter().take(num_sets - 1) {
            size_to_shave *= input_dims[*i as usize] as u64;
            let buff_size: u64 = input_buff.0.size / size_to_shave;
            scratch_buffs.push(self.reduce.inst.create_buffer(
                VultenBufferType::Device,
                buff_size,
                false,
                false,
            ));
        }

        //0 - uniform with strides
        //1 - input
        //.. - scratch
        //-1 - output
        let mut descriptor_buff_infos: Vec<DescriptorBufferInfo> =
            Vec::with_capacity(3 + scratch_buffs.len());

        descriptor_buff_infos.push(
            DescriptorBufferInfo::default()
                .range(self.stride_uniform.size)
                .offset(0)
                .buffer(self.stride_uniform.vk_buffer),
        );
        descriptor_buff_infos.push(
            DescriptorBufferInfo::default()
                .range(input_buff.0.size)
                .offset(input_buff.1)
                .buffer(input_buff.0.vk_buffer),
        );
        for i in &scratch_buffs {
            descriptor_buff_infos.push(
                DescriptorBufferInfo::default()
                    .range(i.size)
                    .offset(0)
                    .buffer(i.vk_buffer),
            );
        }
        descriptor_buff_infos.push(
            DescriptorBufferInfo::default()
                .range(output_buff.0.size)
                .offset(output_buff.1)
                .buffer(output_buff.0.vk_buffer),
        );

        let mut write_sets: Vec<WriteDescriptorSet> = Vec::with_capacity(3 * num_sets);
        for (i, set) in descriptor_sets.iter().enumerate().take(num_sets) {
            write_sets.push(
                WriteDescriptorSet::default()
                    .dst_set(set.descriptor[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&descriptor_buff_infos.as_slice()[i + 1..i + 2]),
            );
            write_sets.push(
                WriteDescriptorSet::default()
                    .dst_set(set.descriptor[0])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&descriptor_buff_infos.as_slice()[i + 2..i + 3]),
            );
            write_sets.push(
                WriteDescriptorSet::default()
                    .dst_set(set.descriptor[0])
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&descriptor_buff_infos.as_slice()[0..1]),
            );
        }
        self.reduce.inst.update_descriptor_sets(&write_sets, &[]);

        self.scratch_buffs = scratch_buffs;
        Ok(descriptor_sets)
    }

    fn record<'b>(
        &mut self,
        mut builder: CommandBufferBuilder<'b>,
        pipeline: Arc<VultenPipeline>,
        descriptors: &[VultenDescriptor],
    ) -> Result<CommandBufferBuilder<'b>, &'static str> {
        let mut push = ReducePushConst::default();

        builder = builder.bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone());

        let input_dims = self.reduce.input_dims.ok_or("Missing input dims")?;
        let total_elements = input_dims.iter().fold(1, |acc, x| acc * *x as u64);

        let barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
            .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

        let mut elements_left = total_elements as i64;
        for (i, axis) in self.reduce.reduce_dims.iter().enumerate() {
            builder = builder.bind_descriptor_sets(
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &descriptors[i].descriptor,
                &[],
            );

            let spec = self.spec.as_ref().ok_or("Missing spec")?;
            let chunk_size =
                self.reduce.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
            elements_left /= input_dims[*axis as usize];
            push.offset = i as u32;
            let chunks = (0..elements_left).as_chunks(chunk_size).into_iter();

            for chunk in chunks {
                push.start = chunk.start as u32;
                push.stop = chunk.end as u32;

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

            builder = builder.pipeline_barrier(
                PipelineStageFlags::COMPUTE_SHADER,
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        Ok(builder)
    }

    fn run(&mut self) -> Result<(), &'static str> {
        let pipeline = self.get_pipeline()?;
        let descriptors = self.get_descriptors(pipeline.clone())?;
        let q = self.reduce.inst.get_queue(QueueFlags::COMPUTE);
        let cmd_buffs = self
            .reduce
            .inst
            .create_cmd_buffers(1, &q)
            .or(Err("Could not create command buffers"))?;
        let builder = CommandBufferBuilder::new(cmd_buffs[0], &self.reduce.inst.device).begin();

        self.record(builder, pipeline, &descriptors)?
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
