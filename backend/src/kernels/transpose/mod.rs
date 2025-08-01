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
    dims::Dims,
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    utills::calculate_strdies,
    VultenDataType, VultenInstance,
};

use super::{Chunkable, KernelBuff};

const TRANSPOSE_SOURCE: &str = include_str!("transpose.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct TransposePipelineSpec {
    local_x: u32,
    input_dims: Vec<i64>,
    transpose: Vec<i64>,
    output_dims: Vec<i64>,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct TransposePushConst {
    start: u32,
    stop: u32,
}

impl PushConstSpec for TransposePushConst {
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

impl PipelineSpec for TransposePipelineSpec {
    type PushConst = TransposePushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new(TRANSPOSE_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.add_define("NUM_DIMS".into(), Some(self.input_dims.len().to_string()));

        let input_stride: Vec<i64> = calculate_strdies(&self.input_dims)
            .into_iter()
            .skip(1)
            .collect();
        let mut tp_stride = Vec::new();
        for val in &self.transpose {
            tp_stride.push(input_stride[*val as usize]);
        }

        let in_stride = calculate_strdies(&self.input_dims);
        let mut in_stride_string = "uint[](".to_string();
        for (idx, val) in in_stride.iter().skip(1).enumerate() {
            let line_end = if idx == in_stride.len() - 2 {
                ")"
            } else {
                ","
            };

            in_stride_string += &(val.to_string() + line_end);
        }
        compiler.add_define("STRIDE_IN".into(), Some(in_stride_string));

        let out_stride = calculate_strdies(&self.output_dims);
        let mut rev_transpose: Vec<usize> = vec![0; self.transpose.len()];
        for i in 0..self.transpose.len(){
            rev_transpose[self.transpose[i] as usize] = i;
        }
        let mut transpose_string = "uint[](".to_string();
        for (idx, val) in rev_transpose.iter().enumerate() {
            let line_end = if idx == rev_transpose.len() - 1 {
                ")"
            } else {
                ","
            };

            transpose_string += &(out_stride[*val+1].to_string() + line_end);
        }
        compiler.add_define("TRANSPOSE".into(), Some(transpose_string));

        compiler.compile().unwrap()
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

pub struct TransposeKernel<'a> {
    inst: &'a VultenInstance,
    d_type: VultenDataType,
    input: Option<KernelBuff<'a>>,
    input_dims: Option<&'a [i64]>,
    transpose: Option<Dims<'a, i64>>,
    output: Option<KernelBuff<'a>>,
    output_dims: Option<Dims<'a, i64>>,
    spec: Option<TransposePipelineSpec>,
}

impl<'a> TransposeKernel<'a> {
    pub fn new(inst: &'a VultenInstance, d_type: VultenDataType) -> Self {
        TransposeKernel {
            inst,
            d_type,
            input: Default::default(),
            input_dims: Default::default(),
            transpose: Default::default(),
            output: Default::default(),
            output_dims: Default::default(),
            spec: Default::default(),
        }
    }

    pub fn input(mut self, buff: KernelBuff<'a>, dims: &'a [i64]) -> Result<Self, &'static str> {
        if dims.contains(&0) {
            return Err("Input has a zero dim!");
        }
        self.input = Some(buff);
        self.input_dims = Some(dims);

        Ok(self)
    }

    pub fn transpose(mut self, axes: Dims<'a, i64>) -> Result<Self, &'static str> {
        self.transpose = Some(axes);

        Ok(self)
    }

    pub fn output(
        mut self,
        buff: KernelBuff<'a>,
        dims: Dims<'a, i64>,
    ) -> Result<Self, &'static str> {
        self.output = Some(buff);
        self.output_dims = Some(dims);

        Ok(self)
    }

    pub fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Transpose(spec.clone())))
        } else {
            let input_dims = self.input_dims.ok_or("Missing input dims")?;
            let transpose = self.transpose.as_ref().ok_or("Missing transpose axes")?;
            let output_dims = self.output_dims.as_ref().ok_or("Missing output dims")?;

            let spec = TransposePipelineSpec {
                local_x: self.inst.device_props.sub_group_size.max(1),
                input_dims: input_dims.to_vec(),
                transpose: transpose.to_vec(),
                output_dims: output_dims.to_vec(),
                d_type: self.d_type,
            };
            let pipeline = self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Transpose(spec.clone()));
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
        let mut push = TransposePushConst {
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

        let spec: &TransposePipelineSpec = self.spec.as_ref().ok_or("Missing spec")?;
        let chunk_size = self.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
        let input_dims = self.input_dims.ok_or("Missing input dims")?;
        let total_elements: i64 = input_dims.iter().product();
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
