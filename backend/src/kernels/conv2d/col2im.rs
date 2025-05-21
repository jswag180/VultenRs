use ash::vk::{
    self, AccessFlags, DependencyFlags, DescriptorType, MemoryBarrier, PipelineBindPoint,
    PipelineStageFlags, PushConstantRange, QueueFlags, ShaderStageFlags, SpecializationInfo,
    SpecializationMapEntry, SubmitInfo, WriteDescriptorSet,
};
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    descriptor::VultenDescriptor,
    kernels::{ChannelFormat, Chunkable, KernelBuff},
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

const COL2IM_SOURCE: &str = include_str!("col2im.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct Col2ImPipelineSpec {
    local_x: u32,
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_h: u32,
    padding_w: u32,
    format: ChannelFormat,
    backprop_dims: [u32; 4],
    filters_dims: [u32; 4],
    output_dims: [u32; 4],
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct Col2ImPushConst {
    pub start: u32,
    pub stop: u32,
    pub offset: u32,
}

impl PushConstSpec for Col2ImPushConst {
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

impl PipelineSpec for Col2ImPipelineSpec {
    type PushConst = Col2ImPushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler = compiler::ShaderCompiler::new(COL2IM_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.add_define("BTACH_SIZE".into(), Some(self.backprop_dims[0].to_string()));

        match self.format {
            ChannelFormat::NHWC => {
                compiler.add_define("FORMAT".into(), Some("0".into()));

                //input
                compiler.add_define(
                    "BACKPROP_HEIGHT".into(),
                    Some(self.backprop_dims[1].to_string()),
                );
                compiler.add_define(
                    "BACKPROP_WIDTH".into(),
                    Some(self.backprop_dims[2].to_string()),
                );
                compiler.add_define("DEPTH_IN".into(), Some(self.backprop_dims[3].to_string()));

                //output
                compiler.add_define(
                    "OUTPUT_HEIGHT".into(),
                    Some(self.output_dims[1].to_string()),
                );
                compiler.add_define("OUTPUT_WIDTH".into(), Some(self.output_dims[2].to_string()));
                compiler.add_define("DEPTH_OUT".into(), Some(self.output_dims[3].to_string()));
            }
            ChannelFormat::NCHW => {
                compiler.add_define("FORMAT".into(), Some("1".into()));

                //input
                compiler.add_define(
                    "BACKPROP_HEIGHT".into(),
                    Some(self.backprop_dims[2].to_string()),
                );
                compiler.add_define(
                    "BACKPROP_WIDTH".into(),
                    Some(self.backprop_dims[3].to_string()),
                );
                compiler.add_define("DEPTH_IN".into(), Some(self.backprop_dims[1].to_string()));

                //output
                compiler.add_define(
                    "OUTPUT_HEIGHT".into(),
                    Some(self.output_dims[2].to_string()),
                );
                compiler.add_define("OUTPUT_WIDTH".into(), Some(self.output_dims[3].to_string()));
                compiler.add_define("DEPTH_OUT".into(), Some(self.output_dims[1].to_string()));
            }
        };

        //filters
        compiler.add_define(
            "FILTER_HEIGHT".into(),
            Some(self.filters_dims[0].to_string()),
        );
        compiler.add_define(
            "FILTER_WIDTH".into(),
            Some(self.filters_dims[1].to_string()),
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
                size: std::mem::size_of_val(&self.stride_h),
            },
            SpecializationMapEntry {
                constant_id: 2,
                offset: 8,
                size: std::mem::size_of_val(&self.stride_w),
            },
            SpecializationMapEntry {
                constant_id: 3,
                offset: 12,
                size: std::mem::size_of_val(&self.dilation_h),
            },
            SpecializationMapEntry {
                constant_id: 4,
                offset: 16,
                size: std::mem::size_of_val(&self.dilation_w),
            },
            SpecializationMapEntry {
                constant_id: 5,
                offset: 20,
                size: std::mem::size_of_val(&self.padding_h),
            },
            SpecializationMapEntry {
                constant_id: 6,
                offset: 24,
                size: std::mem::size_of_val(&self.padding_w),
            },
        ];

        let mut spec_buffer: Vec<u8> = Vec::new();
        let local_x_slice = self.local_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_x_slice);
        let stride_h_slice = self.stride_h.to_ne_bytes();
        spec_buffer.extend_from_slice(&stride_h_slice);
        let stride_w_slice = self.stride_w.to_ne_bytes();
        spec_buffer.extend_from_slice(&stride_w_slice);
        let dilation_h_slice = self.dilation_h.to_ne_bytes();
        spec_buffer.extend_from_slice(&dilation_h_slice);
        let dilation_w_slice = self.dilation_w.to_ne_bytes();
        spec_buffer.extend_from_slice(&dilation_w_slice);
        let padding_h_slice = self.padding_h.to_ne_bytes();
        spec_buffer.extend_from_slice(&padding_h_slice);
        let padding_w_slice = self.padding_w.to_ne_bytes();
        spec_buffer.extend_from_slice(&padding_w_slice);

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

pub struct Col2ImgKernel<'a> {
    inst: &'a VultenInstance,
    d_type: VultenDataType,
    padd_h: u32,
    padd_w: u32,
    stride_x: u32,
    stride_y: u32,
    dilation_x: u32,
    dilation_y: u32,
    format: ChannelFormat,
    filter_dims: Option<&'a [i64]>,
    backprop_dims: Option<&'a [i64]>,
    input: Option<KernelBuff<'a>>,
    output: Option<KernelBuff<'a>>,
    output_dims: Option<&'a [i64]>,
    spec: Option<Col2ImPipelineSpec>,
}

impl<'a> Col2ImgKernel<'a> {
    pub fn new(
        inst: &'a VultenInstance,
        d_type: VultenDataType,
        padding: (u32, u32),
        strides: (u32, u32),
        dilations: (u32, u32),
        format: ChannelFormat,
    ) -> Self {
        Self {
            inst,
            d_type,
            padd_h: padding.0,
            padd_w: padding.0,
            stride_x: strides.0,
            stride_y: strides.0,
            dilation_x: dilations.0,
            dilation_y: dilations.0,
            format,
            filter_dims: Default::default(),
            backprop_dims: Default::default(),
            input: Default::default(),
            output: Default::default(),
            output_dims: Default::default(),
            spec: Default::default(),
        }
    }

    pub fn filter(mut self, dims: &'a [i64]) -> Result<Self, &'static str> {
        self.filter_dims = Some(dims);

        Ok(self)
    }

    pub fn backprop(mut self, dims: &'a [i64]) -> Result<Self, &'static str> {
        self.backprop_dims = Some(dims);

        Ok(self)
    }

    pub fn input(mut self, input: KernelBuff<'a>) -> Result<Self, &'static str> {
        self.input = Some(input);

        Ok(self)
    }

    pub fn output(mut self, output: KernelBuff<'a>, dims: &'a [i64]) -> Result<Self, &'static str> {
        self.output_dims = Some(dims);
        self.output = Some(output);

        Ok(self)
    }

    pub fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Col2Im(spec.clone())))
        } else {
            let backprop_dims = self.backprop_dims.ok_or("Missing backprop dims")?;
            let filters_dims = self.filter_dims.ok_or("Missing filter dims")?;
            let output_dims = self.output_dims.ok_or("Missing output dims")?;
            let spec = Col2ImPipelineSpec {
                local_x: self.inst.device_props.sub_group_size.max(1),
                stride_h: self.stride_x,
                stride_w: self.stride_y,
                dilation_h: self.dilation_x,
                dilation_w: self.dilation_y,
                padding_h: self.padd_h,
                padding_w: self.padd_w,
                format: self.format,
                backprop_dims: [
                    backprop_dims[0] as u32,
                    backprop_dims[1] as u32,
                    backprop_dims[2] as u32,
                    backprop_dims[3] as u32,
                ],
                filters_dims: [
                    filters_dims[0] as u32,
                    filters_dims[1] as u32,
                    filters_dims[2] as u32,
                    filters_dims[3] as u32,
                ],
                output_dims: [
                    output_dims[0] as u32,
                    output_dims[1] as u32,
                    output_dims[2] as u32,
                    output_dims[3] as u32,
                ],
                d_type: self.d_type,
            };
            let pipeline = self
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Col2Im(spec.clone()));
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
        let mut push = Col2ImPushConst::default();
        let backprop_dims = self.backprop_dims.ok_or("Missing backprop dims")?;
        let total_elements: i64 = match self.format {
            ChannelFormat::NHWC => backprop_dims[0] * backprop_dims[1] * backprop_dims[2],
            ChannelFormat::NCHW => backprop_dims[0] * backprop_dims[2] * backprop_dims[3],
        };

        let zero_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);
        let step_barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
            .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

        let output_desc_buff = self
            .output
            .as_ref()
            .ok_or("No output operand")?
            .get_descriptor_info()?;
        builder = builder
            .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &descriptors.descriptor,
                &[],
            )
            .fill_buffer(
                output_desc_buff[0].buffer,
                output_desc_buff[0].offset,
                output_desc_buff[0].range,
                0,
            )
            .pipeline_barrier(
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(),
                &[zero_barrier],
                &[],
                &[],
            );

        let spec = self.spec.as_ref().ok_or("Missing spec")?;
        let chunk_size = self.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
        let filters_dims = self.filter_dims.ok_or("Missing filter dims")?;
        let filter_vol = filters_dims[0] * filters_dims[1];
        for i in 0..filter_vol {
            let chunks = (0..total_elements).as_chunks(chunk_size).into_iter();

            for chunk in chunks {
                push.start = chunk.start as u32;
                push.stop = chunk.end as u32;
                push.offset = i as u32;

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
                &[step_barrier],
                &[],
                &[],
            );
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
