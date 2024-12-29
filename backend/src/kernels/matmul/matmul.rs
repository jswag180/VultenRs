use ash::vk::{
    self, AccessFlags, DependencyFlags, DescriptorType, MemoryBarrier, PipelineBindPoint,
    PipelineStageFlags, QueueFlags, ShaderStageFlags, SubmitInfo, WriteDescriptorSet,
};
use std::rc::Rc;

use crate::{
    cmd_buff::CommandBufferBuilder,
    kernels::{Chunkable, KernelInput},
    pipeline::{PipelineSpecs, PushConstSpec},
    VultenDataType, VultenInstance,
};

use super::{
    get_block_dims,
    transpose::{TransposePipelineSpec, TransposePushConst},
    MatmulPipelineSpec, MatmulPushConst, BROADCAST_NONE,
};

pub fn run(
    inst: &VultenInstance,
    d_type: VultenDataType,
    a: &KernelInput,
    trans_a: bool,
    b: &KernelInput,
    trans_b: bool,
    output: &KernelInput,
) -> Result<(), &'static str> {
    let mat_a_post: (i64, i64) = if trans_a {
        (a.dims[1], a.dims[0])
    } else {
        (a.dims[0], a.dims[1])
    };
    let mat_b_post: (i64, i64) = if trans_b {
        (b.dims[1], b.dims[0])
    } else {
        (b.dims[0], b.dims[1])
    };
    let block_dims = get_block_dims(mat_a_post, mat_b_post);
    let num_blocks_x = (mat_a_post.0 as f32 / block_dims.0 as f32).ceil() as i64;
    let num_blocks_y = (mat_b_post.1 as f32 / block_dims.1 as f32).ceil() as i64;

    let matmul_spec = MatmulPipelineSpec {
        local_x: inst.device_props.sub_group_size.max(1),
        block_size_x: block_dims.0,
        block_size_y: block_dims.1,
        bk_cont: mat_a_post.1 as u32 / block_dims.0,
        a_x: mat_a_post.0 as u32,
        a_y: mat_a_post.1 as u32,
        b_x: mat_b_post.0 as u32,
        b_y: mat_b_post.1 as u32,
        inline_trans_a: false,
        inline_trans_b: false,
        bk_num_y: num_blocks_y as u32,
        broadcast: BROADCAST_NONE,
        d_type,
    };
    let matmul_pipeline = inst.get_pipeline_from_spec(PipelineSpecs::Matmul(matmul_spec.clone()));

    let matmul_descriptors = inst
        .get_descriptor_set(DescriptorType::STORAGE_BUFFER, matmul_pipeline.clone())
        .unwrap();

    let trans_spec = TransposePipelineSpec {
        local_x: inst.device_props.sub_group_size.max(1),
        d_type: d_type,
    };
    let trans_pipeline = inst.get_pipeline_from_spec(PipelineSpecs::Transpose(trans_spec.clone()));

    let a_desc_buff = a.buff.get_descriptor_info()?;
    let b_desc_buff = b.buff.get_descriptor_info()?;
    let output_desc_buff = output.buff.get_descriptor_info()?;

    let a_trans_buff = if trans_a {
        let buff = Rc::new(inst.create_buffer(
            crate::memory::VultenBufferType::Device,
            a_desc_buff[0].range,
            false,
            false,
        ));
        Some((
            inst.get_descriptor_set(DescriptorType::STORAGE_BUFFER, trans_pipeline.clone())
                .unwrap(),
            [vk::DescriptorBufferInfo::default()
                .buffer(buff.vk_buffer)
                .range(buff.size)
                .offset(0)],
            buff,
        ))
    } else {
        None
    };
    let b_trans_buff = if trans_b {
        let buff = Rc::new(inst.create_buffer(
            crate::memory::VultenBufferType::Device,
            b_desc_buff[0].range,
            false,
            false,
        ));
        Some((
            inst.get_descriptor_set(DescriptorType::STORAGE_BUFFER, trans_pipeline.clone())
                .unwrap(),
            [vk::DescriptorBufferInfo::default()
                .buffer(buff.vk_buffer)
                .range(buff.size)
                .offset(0)],
            buff,
        ))
    } else {
        None
    };

    //The most we will ever need a 7(3+2+2)
    let mut write_sets: Vec<WriteDescriptorSet> = Vec::with_capacity(7);
    if let Some((desc, desc_buff, _buff)) = a_trans_buff.as_ref() {
        write_sets.push(
            WriteDescriptorSet::default()
                .dst_set(desc.descriptor[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&a_desc_buff),
        );

        write_sets.push(
            WriteDescriptorSet::default()
                .dst_set(desc.descriptor[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(desc_buff),
        );

        write_sets.push(
            WriteDescriptorSet::default()
                .dst_set(matmul_descriptors.descriptor[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(desc_buff),
        );
    } else {
        write_sets.push(
            WriteDescriptorSet::default()
                .dst_set(matmul_descriptors.descriptor[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&a_desc_buff),
        );
    }

    if let Some((desc, desc_buff, _buff)) = b_trans_buff.as_ref() {
        write_sets.push(
            WriteDescriptorSet::default()
                .dst_set(desc.descriptor[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&b_desc_buff),
        );

        write_sets.push(
            WriteDescriptorSet::default()
                .dst_set(desc.descriptor[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(desc_buff),
        );

        write_sets.push(
            WriteDescriptorSet::default()
                .dst_set(matmul_descriptors.descriptor[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(desc_buff),
        );
    } else {
        write_sets.push(
            WriteDescriptorSet::default()
                .dst_set(matmul_descriptors.descriptor[0])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER)
                .buffer_info(&b_desc_buff),
        );
    }

    write_sets.push(
        WriteDescriptorSet::default()
            .dst_set(matmul_descriptors.descriptor[0])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&output_desc_buff),
    );
    inst.update_descriptor_sets(&write_sets, &[]);

    let q = inst.get_queue(QueueFlags::COMPUTE);
    let cmd_buffs = inst.create_cmd_buffers(1, &q).unwrap();

    let mut matmul_push = MatmulPushConst {
        start_x: 0,
        stop_x: (num_blocks_x * num_blocks_y) as u32,
        offset: 0,
    };

    let mut trans_push = TransposePushConst {
        start: 0,
        stop: 0,
        hight: 0,
        width: 0,
    };

    let transpose_barrier = MemoryBarrier::default()
        .src_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

    let mut builder = CommandBufferBuilder::new(cmd_buffs[0], &inst.device).begin();

    if let Some((desc, _desc_buff, _buff)) = a_trans_buff.as_ref() {
        builder = builder
            .bind_pipeline(PipelineBindPoint::COMPUTE, trans_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::COMPUTE,
                trans_pipeline.pipeline_layout,
                0,
                &desc.descriptor,
                &[],
            );

        let a_total_elements = a.dims[0] * a.dims[1];
        let work_window = inst.device_props.max_work_group[0] as i64 * trans_spec.local_x as i64;
        trans_push.hight = a.dims[0] as u32;
        trans_push.width = a.dims[1] as u32;
        if a_total_elements > work_window {
            let windows = (0..a_total_elements).as_chunks(work_window);
            for window in windows {
                trans_push.start = window.start as u32;
                trans_push.stop = window.end as u32;

                let threads =
                    ((window.end - window.start) as f32 / trans_spec.local_x as f32).ceil() as u32;
                builder = builder
                    .push_constants(
                        trans_pipeline.pipeline_layout,
                        ShaderStageFlags::COMPUTE,
                        0,
                        trans_push.get_slice(),
                    )
                    .dispatch(threads, 1, 1);
            }
        } else {
            let a_total_elements = a.dims[0] * a.dims[1];
            trans_push.start = 0;
            trans_push.stop = a_total_elements as u32;

            let threads = (a_total_elements as f32 / trans_spec.local_x as f32).ceil() as u32;
            builder = builder
                .push_constants(
                    trans_pipeline.pipeline_layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    trans_push.get_slice(),
                )
                .dispatch(threads, 1, 1);
        }
    }

    if let Some((desc, _desc_buff, _buff)) = b_trans_buff.as_ref() {
        builder = builder
            .bind_pipeline(PipelineBindPoint::COMPUTE, trans_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::COMPUTE,
                trans_pipeline.pipeline_layout,
                0,
                &desc.descriptor,
                &[],
            );

        let b_total_elements = b.dims[0] * b.dims[1];
        let work_window = inst.device_props.max_work_group[0] as i64 * trans_spec.local_x as i64;
        trans_push.hight = b.dims[0] as u32;
        trans_push.width = b.dims[1] as u32;
        if b_total_elements > work_window {
            let windows = (0..b_total_elements).as_chunks(work_window);
            for window in windows {
                trans_push.start = window.start as u32;
                trans_push.stop = window.end as u32;

                let threads =
                    ((window.end - window.start) as f32 / trans_spec.local_x as f32).ceil() as u32;
                builder = builder
                    .push_constants(
                        trans_pipeline.pipeline_layout,
                        ShaderStageFlags::COMPUTE,
                        0,
                        trans_push.get_slice(),
                    )
                    .dispatch(threads, 1, 1);
            }
        } else {
            let b_total_elements = b.dims[0] * b.dims[1];
            trans_push.start = 0;
            trans_push.stop = b_total_elements as u32;

            let threads = (b_total_elements as f32 / trans_spec.local_x as f32).ceil() as u32;
            builder = builder
                .push_constants(
                    trans_pipeline.pipeline_layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    trans_push.get_slice(),
                )
                .dispatch(threads, 1, 1);
        }
    }

    if trans_a || trans_b {
        builder = builder.pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            &[transpose_barrier],
            &[],
            &[],
        );
    }

    builder = builder
        .bind_pipeline(PipelineBindPoint::COMPUTE, matmul_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::COMPUTE,
            matmul_pipeline.pipeline_layout,
            0,
            &matmul_descriptors.descriptor,
            &[],
        );

    let chunk_size: i64 = inst.device_props.max_work_group[0] as i64 * matmul_spec.local_x as i64;
    if num_blocks_x * num_blocks_y > chunk_size {
        let chunks = (0..num_blocks_x * num_blocks_y)
            .as_chunks(chunk_size)
            .into_iter();

        for chunk in chunks {
            matmul_push.start_x = chunk.start as u32 * 1;
            matmul_push.stop_x = chunk.end as u32;

            let threads = matmul_push.stop_x - matmul_push.start_x;
            builder = builder
                .push_constants(
                    matmul_pipeline.pipeline_layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    matmul_push.get_slice(),
                )
                .dispatch(threads, 1, 1);
        }
    } else {
        let threads =
            ((num_blocks_x * num_blocks_y) as f32 / matmul_spec.local_x as f32).ceil() as u32;
        builder = builder
            .push_constants(
                matmul_pipeline.pipeline_layout,
                ShaderStageFlags::COMPUTE,
                0,
                matmul_push.get_slice(),
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
