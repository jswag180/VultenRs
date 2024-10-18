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
    kernels::{ChannelFormat, Chunkable, KernelInput},
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    va::VaAddress,
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

#[derive(Debug, AsBytes)]
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

    fn get_shader(&self) -> CompilationArtifact {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new("im2col.comp", COL2IM_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler
            .opts
            .add_macro_definition("BTACH_SIZE", Some(&self.backprop_dims[0].to_string()));

        match self.format {
            ChannelFormat::NHWC => {
                compiler.opts.add_macro_definition("FORMAT", Some("0"));

                //input
                compiler.opts.add_macro_definition(
                    "BACKPROP_HEIGHT",
                    Some(&self.backprop_dims[1].to_string()),
                );
                compiler.opts.add_macro_definition(
                    "BACKPROP_WIDTH",
                    Some(&self.backprop_dims[2].to_string()),
                );
                compiler
                    .opts
                    .add_macro_definition("DEPTH_IN", Some(&self.backprop_dims[3].to_string()));

                //output
                compiler
                    .opts
                    .add_macro_definition("OUTPUT_HEIGHT", Some(&self.output_dims[1].to_string()));
                compiler
                    .opts
                    .add_macro_definition("OUTPUT_WIDTH", Some(&self.output_dims[2].to_string()));
                compiler
                    .opts
                    .add_macro_definition("DEPTH_OUT", Some(&self.output_dims[3].to_string()));
            }
            ChannelFormat::NCHW => {
                compiler.opts.add_macro_definition("FORMAT", Some("1"));

                //input
                compiler.opts.add_macro_definition(
                    "BACKPROP_HEIGHT",
                    Some(&self.backprop_dims[2].to_string()),
                );
                compiler.opts.add_macro_definition(
                    "BACKPROP_WIDTH",
                    Some(&self.backprop_dims[3].to_string()),
                );
                compiler
                    .opts
                    .add_macro_definition("DEPTH_IN", Some(&self.backprop_dims[1].to_string()));

                //output
                compiler
                    .opts
                    .add_macro_definition("OUTPUT_HEIGHT", Some(&self.output_dims[2].to_string()));
                compiler
                    .opts
                    .add_macro_definition("OUTPUT_WIDTH", Some(&self.output_dims[3].to_string()));
                compiler
                    .opts
                    .add_macro_definition("DEPTH_OUT", Some(&self.output_dims[1].to_string()));
            }
        };

        //filters
        compiler
            .opts
            .add_macro_definition("FILTER_HEIGHT", Some(&self.filters_dims[0].to_string()));
        compiler
            .opts
            .add_macro_definition("FILTER_WIDTH", Some(&self.filters_dims[1].to_string()));

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
    padding: (u32, u32),
    format: ChannelFormat,
    strides: (u32, u32),
    dilations: (u32, u32),
    filters_dims: &[i64],
    backprop_dims: &[i64],
    input: VaAddress,
    output: KernelInput,
) -> Result<(), &'static str> {
    let spec = Col2ImPipelineSpec {
        local_x: inst.device_props.sub_group_size.max(1),
        stride_h: strides.0,
        stride_w: strides.1,
        dilation_h: dilations.0,
        dilation_w: dilations.1,
        padding_h: padding.0,
        padding_w: padding.1,
        format,
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
            output.dims[0] as u32,
            output.dims[1] as u32,
            output.dims[2] as u32,
            output.dims[3] as u32,
        ],
        d_type,
    };
    let pipeline = inst.get_pipeline_from_spec(PipelineSpecs::Col2Im(spec.clone()));

    let descriptors = inst
        .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
        .unwrap();

    let input_desc_buff = VultenInstance::get_descriptor_info_va(input).unwrap();
    let output_desc_buff = VultenInstance::get_descriptor_info_va(output.addr).unwrap();

    let write_sets = [
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&input_desc_buff.0),
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&output_desc_buff.0),
    ];
    inst.update_descriptor_sets(&write_sets, &[]);

    let q = inst.get_queue(QueueFlags::COMPUTE);
    let cmd_buffs = inst.create_cmd_buffers(1, &q).unwrap();

    let total_elements: i64 = match format {
        ChannelFormat::NHWC => backprop_dims[0] * backprop_dims[1] * backprop_dims[2],
        ChannelFormat::NCHW => backprop_dims[0] * backprop_dims[2] * backprop_dims[3],
    };
    let mut push = Col2ImPushConst {
        start: 0,
        stop: total_elements as u32,
        offset: 0,
    };

    let zero_barrier = MemoryBarrier::default()
        .src_access_mask(AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);
    let step_barrier = MemoryBarrier::default()
        .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

    let mut builder = CommandBufferBuilder::new(cmd_buffs[0], &inst.device)
        .begin()
        .bind_pipeline(PipelineBindPoint::COMPUTE, pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::COMPUTE,
            pipeline.pipeline_layout,
            0,
            &descriptors.descriptor,
            &[],
        )
        .fill_buffer(output_desc_buff.1, 0, output_desc_buff.0[0].range, 0)
        .pipeline_barrier(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            &[zero_barrier],
            &[],
            &[],
        );

    let chunk_size = inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
    let filter_vol = filters_dims[0] * filters_dims[1] * filters_dims[2];
    if total_elements as i64 > chunk_size {
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
    } else {
        for i in 0..filter_vol {
            push.offset = i as u32;

            let threads = (total_elements as f32 / spec.local_x as f32).ceil() as u32;

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
                &[step_barrier],
                &[],
                &[],
            );
        }
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