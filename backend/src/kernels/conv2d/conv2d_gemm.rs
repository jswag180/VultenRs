use ash::vk::{
    self, DescriptorType, PipelineBindPoint, PushConstantRange, QueueFlags, ShaderStageFlags,
    SpecializationInfo, SpecializationMapEntry, SubmitInfo, WriteDescriptorSet,
};
use shaderc::CompilationArtifact;
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    kernels::{Chunkable, KernelInput},
    pipeline::{PipelineSpec, PipelineSpecs, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

const CONV2D_GEMM_SOURCE: &str = include_str!("conv2d_gemm.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct Conv2DGemmPipelineSpec {
    local_x: u32,
    cols_dims: [u32; 3],
    filters_dims: [u32; 4],
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes)]
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

    fn get_shader(&self) -> CompilationArtifact {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new("conv2d_gemm.comp", CONV2D_GEMM_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler
            .opts
            .add_macro_definition("COLS_BATCH", Some(&self.cols_dims[0].to_string()));
        compiler
            .opts
            .add_macro_definition("COLS_HEIGHT", Some(&self.cols_dims[1].to_string()));
        compiler
            .opts
            .add_macro_definition("COLS_WIDTH", Some(&self.cols_dims[2].to_string()));

        //filters
        compiler
            .opts
            .add_macro_definition("FILTER_HEIGHT", Some(&self.filters_dims[0].to_string()));
        compiler
            .opts
            .add_macro_definition("FILTER_WIDTH", Some(&self.filters_dims[1].to_string()));
        compiler
            .opts
            .add_macro_definition("DEPTH_IN", Some(&self.filters_dims[2].to_string()));
        compiler
            .opts
            .add_macro_definition("DEPTH_OUT", Some(&self.filters_dims[3].to_string()));

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
        let desc_types: Vec<vk::DescriptorType> = vec![vk::DescriptorType::STORAGE_BUFFER; 3];
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
