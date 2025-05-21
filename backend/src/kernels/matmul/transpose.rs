use ash::vk::{
    self, DescriptorType, PipelineBindPoint, PushConstantRange, ShaderStageFlags,
    SpecializationInfo, SpecializationMapEntry, WriteDescriptorSet,
};
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    descriptor::VultenDescriptor,
    kernels::{Chunkable, KernelBuff},
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

const TRANSPOSE_SOURCE: &str = include_str!("transpose.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct TransposePipelineSpec {
    pub local_x: u32,
    pub d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct TransposePushConst {
    pub start: u32,
    pub stop: u32,
    pub hight: u32,
    pub width: u32,
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
        let slice: &[u8; 16] = zerocopy::transmute_ref!(self);

        slice
    }
}

impl PipelineSpec for TransposePipelineSpec {
    type PushConst = TransposePushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new(TRANSPOSE_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

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
    output: Option<KernelBuff<'a>>,
    spec: Option<TransposePipelineSpec>,
}

impl<'a> TransposeKernel<'a> {
    pub fn new(inst: &'a VultenInstance, d_type: VultenDataType) -> Self {
        Self {
            inst,
            d_type,
            input: Default::default(),
            input_dims: Default::default(),
            output: Default::default(),
            spec: Default::default(),
        }
    }

    pub fn input(
        &mut self,
        input: KernelBuff<'a>,
        dims: &'a [i64],
    ) -> Result<&mut Self, &'static str> {
        if dims.iter().product::<i64>() > u32::MAX as i64 {
            return Err("Input is to large!");
        }

        self.input_dims = Some(dims);
        self.input = Some(input);

        Ok(self)
    }

    pub fn output(&mut self, output: KernelBuff<'a>) -> Result<&mut Self, &'static str> {
        self.output = Some(output);

        Ok(self)
    }

    pub fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Transpose(spec.clone())))
        } else {
            let spec = TransposePipelineSpec {
                local_x: self.inst.device_props.sub_group_size.max(1),
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
        let mut push = TransposePushConst::default();

        builder = builder
            .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &descriptors.descriptor,
                &[],
            );

        let dims = self.input_dims.ok_or("Missing input dims")?;
        let batches;
        if dims.len() == 2 {
            batches = 1;
            push.hight = dims[0] as u32;
            push.width = dims[1] as u32;
        } else {
            batches = dims[0] as u32;
            push.hight = dims[1] as u32;
            push.width = dims[2] as u32;
        }
        let total_elements = (push.hight * push.width) as i64;
        let spec = self.spec.as_ref().ok_or("Missing spec")?;
        let work_window = self.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
        let windows = (0..total_elements).as_chunks(work_window);
        for batch in 0..batches {
            for window in &windows {
                push.start = (total_elements as u32 * batch) + window.start as u32;
                push.stop = push.start + window.end as u32;

                let threads =
                    ((window.end - window.start) as f32 / spec.local_x as f32).ceil() as u32;
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
}
