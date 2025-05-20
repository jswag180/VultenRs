use ash::vk::{
    self, AccessFlags, DependencyFlags, DescriptorType, DeviceSize, MemoryBarrier,
    PipelineBindPoint, PipelineStageFlags, PushConstantRange, QueueFlags, ShaderStageFlags,
    SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet,
};
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    dims::Dims,
    kernels::{ChannelFormat, Chunkable, KernelBuff, KernelInput},
    memory::VultenBuffer,
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

use super::{im2col::Img2ColKernel, Conv2DKernel, Conv2DKernelVersion};

const CONV2D_GEMM_SOURCE: &str = include_str!("conv2d_gemm.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct Conv2DGemmPipelineSpec {
    local_x: u32,
    cols_dims: [u32; 3],
    filters_dims: [u32; 4],
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct Conv2DGemmPushConst {
    pub start: u32,
    pub stop: u32,
}

impl PushConstSpec for Conv2DGemmPushConst {
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

impl PipelineSpec for Conv2DGemmPipelineSpec {
    type PushConst = Conv2DGemmPushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new(CONV2D_GEMM_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.add_define("COLS_BATCH".into(), Some(self.cols_dims[0].to_string()));
        compiler.add_define("COLS_HEIGHT".into(), Some(self.cols_dims[1].to_string()));
        compiler.add_define("COLS_WIDTH".into(), Some(self.cols_dims[2].to_string()));

        //filters
        compiler.add_define(
            "FILTER_HEIGHT".into(),
            Some(self.filters_dims[0].to_string()),
        );
        compiler.add_define(
            "FILTER_WIDTH".into(),
            Some(self.filters_dims[1].to_string()),
        );
        compiler.add_define("DEPTH_IN".into(), Some(self.filters_dims[2].to_string()));
        compiler.add_define("DEPTH_OUT".into(), Some(self.filters_dims[3].to_string()));

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
        let desc_types: Vec<vk::DescriptorType> = vec![vk::DescriptorType::STORAGE_BUFFER; 3];
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

pub fn run(
    inst: &VultenInstance,
    d_type: VultenDataType,
    input: &KernelInput,
    filters: &KernelInput,
    output: &KernelInput,
) -> Result<(), &'static str> {
    let spec = Conv2DGemmPipelineSpec {
        local_x: inst.device_props.sub_group_size.max(1),
        cols_dims: [
            input.dims[0] as u32,
            input.dims[1] as u32,
            input.dims[2] as u32,
        ],
        filters_dims: [
            filters.dims[0] as u32,
            filters.dims[1] as u32,
            filters.dims[2] as u32,
            filters.dims[3] as u32,
        ],
        d_type,
    };
    let pipeline = inst.get_pipeline_from_spec(PipelineSpecs::Conv2DGemm(spec.clone()));

    let descriptors = inst
        .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
        .unwrap();

    let input_desc_buff = input.buff.get_descriptor_info()?;
    let filters_desc_buff = filters.buff.get_descriptor_info()?;
    let output_desc_buff = output.buff.get_descriptor_info()?;

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
            .buffer_info(&filters_desc_buff),
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

    let total_elements: i64 = input.dims[0] * input.dims[1] * filters.dims[3];
    let mut push = Conv2DGemmPushConst {
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

pub struct Conv2DKernelGemm<'a> {
    conv2d: Conv2DKernel<'a>,
    im2col: Img2ColKernel<'a>,
    //im2col_dims: Vec<i64>,
    im2col_buff: Arc<VultenBuffer<'a>>,
    spec: Option<Conv2DGemmPipelineSpec>,
}

impl<'a> Conv2DKernelGemm<'a> {
    pub fn new(conv2d: Conv2DKernel<'a>) -> Result<Self, &'static str> {
        let input = conv2d.input.as_ref().ok_or("Missing input")?;
        let input_dims = conv2d.input_dims.as_ref().ok_or("Missing input dims")?;
        let output_dims = conv2d.output_dims.as_ref().ok_or("Missing output dims")?;
        let filter_dims = conv2d.filter_dims.as_ref().ok_or("Missing filter dims")?;

        let im2col_dims: Vec<i64> = match conv2d.format {
            ChannelFormat::NHWC => vec![
                output_dims[0],
                output_dims[1],
                output_dims[2],
                filter_dims[2] as i64,
            ],
            ChannelFormat::NCHW => vec![
                output_dims[0],
                filter_dims[2] as i64,
                output_dims[2],
                output_dims[3],
            ],
        };

        let size = conv2d.d_type.size_of()? as i64
            * im2col_dims.iter().product::<i64>()
            * filter_dims[0] as i64
            * filter_dims[1] as i64;
        let im2col_buff = Arc::new(conv2d.inst.create_buffer(
            crate::memory::VultenBufferType::Device,
            size as DeviceSize,
            false,
            false,
        ));

        let im2col = Img2ColKernel::new(
            conv2d.inst,
            conv2d.d_type,
            (conv2d.padd_h, conv2d.padd_w),
            (conv2d.stride_x, conv2d.stride_y),
            (conv2d.dilation_x, conv2d.dilation_y),
            conv2d.format,
        )
        .filter(Dims::Slice(filter_dims))?
        .input(input.clone(), Dims::Slice(input_dims))?
        .output(
            crate::kernels::KernelBuff::Buff(im2col_buff.clone()),
            Dims::Vec(im2col_dims),
        )?;

        Ok(Self {
            conv2d,
            im2col,
            //im2col_dims,
            im2col_buff,
            spec: Default::default(),
        })
    }
}

impl<'a> Conv2DKernelVersion<'a> for Conv2DKernelGemm<'a> {
    fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str> {
        if let Some(spec) = self.spec.as_ref() {
            Ok(self
                .conv2d
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Conv2DGemm(spec.clone())))
        } else {
            let input_dims = self
                .conv2d
                .input_dims
                .as_ref()
                .ok_or("Missing input dims")?;
            let filter_dims = self
                .conv2d
                .filter_dims
                .as_ref()
                .ok_or("Missing filter dims")?;
            let output_dims = self
                .conv2d
                .output_dims
                .as_ref()
                .ok_or("Missing output dims")?;
            let output_vol = match self.conv2d.format {
                ChannelFormat::NHWC => output_dims[1] * output_dims[2],
                ChannelFormat::NCHW => output_dims[2] * output_dims[3],
            } as u32;
            let cols_dims = [
                input_dims[0] as u32,
                output_vol,
                (filter_dims[0] * filter_dims[1] * filter_dims[2]) as u32,
            ];
            let spec = Conv2DGemmPipelineSpec {
                local_x: self.conv2d.inst.device_props.sub_group_size.max(1),
                cols_dims,
                filters_dims: [
                    filter_dims[0] as u32,
                    filter_dims[1] as u32,
                    filter_dims[2] as u32,
                    filter_dims[3] as u32,
                ],
                d_type: self.conv2d.d_type,
            };

            let pipeline = self
                .conv2d
                .inst
                .get_pipeline_from_spec(PipelineSpecs::Conv2DGemm(spec.clone()));
            self.spec = Some(spec);

            Ok(pipeline)
        }
    }

    fn get_descriptors(
        &mut self,
        pipeline: Arc<VultenPipeline>,
    ) -> Result<Vec<crate::descriptor::VultenDescriptor<'a>>, &'static str> {
        let mut descriptors = Vec::new();
        let gemm_descriptor = self
            .conv2d
            .inst
            .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline)
            .or(Err("Could not get descriptor set"))?;
        descriptors.push(gemm_descriptor);

        let input_desc_buff = KernelBuff::Buff(self.im2col_buff.clone()).get_descriptor_info()?;
        let filters_desc_buff = self
            .conv2d
            .filters
            .as_ref()
            .ok_or("No b operand")?
            .get_descriptor_info()?;
        let output_desc_buff = self
            .conv2d
            .output
            .as_ref()
            .ok_or("No output operand")?
            .get_descriptor_info()?;

        let write_sets = [
            WriteDescriptorSet::default()
                .dst_set(descriptors[0].descriptor[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&input_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors[0].descriptor[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&filters_desc_buff),
            WriteDescriptorSet::default()
                .dst_set(descriptors[0].descriptor[0])
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_desc_buff),
        ];
        self.conv2d.inst.update_descriptor_sets(&write_sets, &[]);

        let im2col_pipeline = self.im2col.get_pipeline()?;
        let im2col_descriptor = self.im2col.get_descriptors(im2col_pipeline)?;
        descriptors.push(im2col_descriptor);

        Ok(descriptors)
    }

    fn record<'b>(
        &mut self,
        mut builder: CommandBufferBuilder<'b>,
        pipeline: Arc<VultenPipeline>,
        descriptors: &[crate::descriptor::VultenDescriptor],
    ) -> Result<CommandBufferBuilder<'b>, &'static str> {
        let im2col_pipeline = self.im2col.get_pipeline()?;
        builder = self
            .im2col
            .record(builder, im2col_pipeline, &descriptors[1])?;
        let barrier = MemoryBarrier::default()
            .src_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
            .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);
        builder = builder.pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );

        builder = builder
            .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &descriptors[0].descriptor,
                &[],
            );
        let mut push = Conv2DGemmPushConst::default();
        let output_dims = self
            .conv2d
            .output_dims
            .as_ref()
            .ok_or("Missing output dims")?;
        let filter_dims = self
            .conv2d
            .filter_dims
            .as_ref()
            .ok_or("Missing filter dims")?;
        let total_elements: i64 = match self.conv2d.format {
            ChannelFormat::NHWC => {
                output_dims[0] * output_dims[1] * output_dims[2] * filter_dims[3] as i64
            }
            ChannelFormat::NCHW => {
                output_dims[0] * output_dims[2] * output_dims[3] * filter_dims[3] as i64
            }
        };
        let spec = self.spec.as_ref().ok_or("Missing spec")?;
        let chunk_size =
            self.conv2d.inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
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

    fn run(&mut self) -> Result<(), &'static str> {
        let pipeline = self.get_pipeline()?;
        let descriptors = self.get_descriptors(pipeline.clone())?;
        let q = self.conv2d.inst.get_queue(QueueFlags::COMPUTE);
        let cmd_buffs = self
            .conv2d
            .inst
            .create_cmd_buffers(1, &q)
            .or(Err("Could not create command buffers"))?;
        let builder = CommandBufferBuilder::new(cmd_buffs[0], &self.conv2d.inst.device).begin();

        self.record(builder, pipeline, &descriptors)?
            .end()
            .build()?;

        let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
        let fence = self
            .conv2d
            .inst
            .create_fence()
            .or(Err("Could not create fence"))?;

        self.conv2d
            .inst
            .submit_queue(&q, &[sub_info], fence)
            .or(Err("Could not submit queue"))?;
        self.conv2d
            .inst
            .wait_for_fences(&[fence], true)
            .or(Err("Fence timed out"))?;

        self.conv2d.inst.destroy_fence(fence);
        self.conv2d.inst.free_cmd_buffers(&q, cmd_buffs);

        Ok(())
    }
}
