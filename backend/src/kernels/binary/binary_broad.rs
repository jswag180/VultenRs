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
    utills::calculate_strdies,
    VultenDataType, VultenInstance,
};

const BINARY_BROAD_SOURCE: &str = include_str!("binary_broad.comp");

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct BinaryBroadPipelineSpec {
    local_x: u32,
    op: super::BinaryOp,
    x_dims: [i64; 9],
    y_dims: [i64; 9],
    strides: [i64; 9],
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes)]
#[repr(C, packed)]
pub struct BinarayBroadPushConst {
    start: u32,
    stop: u32,
}

impl PushConstSpec for BinarayBroadPushConst {
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

impl PipelineSpec for BinaryBroadPipelineSpec {
    type PushConst = BinarayBroadPushConst;

    fn get_shader(&self) -> CompilationArtifact {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new("binary_broad.comp", BINARY_BROAD_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        for (i, (dim_x, dim_y)) in self.x_dims.iter().zip(self.y_dims.iter()).enumerate() {
            compiler
                .opts
                .add_macro_definition(&format!("DIM_X_{:}", i), Some(&dim_x.to_string()));
            compiler
                .opts
                .add_macro_definition(&format!("DIM_Y_{:}", i), Some(&dim_y.to_string()));
        }

        let strides_str = self.strides.iter().fold("".to_string(), |mut acc, x| {
            acc += &(x.to_string() + ",");
            acc
        });
        compiler
            .opts
            .add_macro_definition("STRIDES_ARR", Some(strides_str.strip_suffix(',').unwrap()));

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
                size: std::mem::size_of::<u32>(),
            },
        ];

        let mut spec_buffer: Vec<u8> = Vec::new();
        let local_x_slice = self.local_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_x_slice);
        let op_as_u32: u32 = self.op.clone().into();
        let op_slice = op_as_u32.to_ne_bytes();
        spec_buffer.extend_from_slice(&op_slice);

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
    op: super::BinaryOp,
    x: KernelInput,
    y: KernelInput,
    output: KernelInput,
) -> Result<(), &'static str> {
    let mut x_dims: [i64; 9] = [1; 9];
    x_dims[9 - x.dims.len()..].clone_from_slice(x.dims);

    let mut y_dims: [i64; 9] = [1; 9];
    y_dims[9 - y.dims.len()..].clone_from_slice(y.dims);

    let mut out_dims: [i64; 9] = [1; 9];
    out_dims[9 - output.dims.len()..].clone_from_slice(output.dims);

    let mut strides_padded: [i64; 9] = [1; 9];
    let strides = calculate_strdies(&out_dims);
    strides_padded.clone_from_slice(&strides[1..]);

    let spec = BinaryBroadPipelineSpec {
        local_x: inst.device_props.sub_group_size.max(1),
        op,
        x_dims,
        y_dims,
        strides: strides_padded,
        d_type,
    };
    let pipeline = inst.get_pipeline_from_spec(PipelineSpecs::BinaryBroad(spec.clone().into()));

    let descriptors = inst
        .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
        .unwrap();

    let x_desc_buff = VultenInstance::get_descriptor_info_va(x.addr).unwrap();
    let y_desc_buff = VultenInstance::get_descriptor_info_va(y.addr).unwrap();
    let output_desc_buff = VultenInstance::get_descriptor_info_va(output.addr).unwrap();

    let write_sets = [
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&x_desc_buff.0),
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&y_desc_buff.0),
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&output_desc_buff.0),
    ];
    inst.update_descriptor_sets(&write_sets, &[]);

    let q = inst.get_queue(QueueFlags::COMPUTE);
    let cmd_buffs = inst.create_cmd_buffers(1, &q).unwrap();

    let total_elements: i64 = out_dims.iter().product();

    let mut push = BinarayBroadPushConst {
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
