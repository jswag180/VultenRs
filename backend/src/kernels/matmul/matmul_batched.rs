use ash::vk::{
    DescriptorType, PipelineBindPoint, QueueFlags, ShaderStageFlags, SubmitInfo, WriteDescriptorSet,
};

use crate::{
    cmd_buff::CommandBufferBuilder,
    kernels::{Chunkable, KernelInput},
    pipeline::{PipelineSpecs, PushConstSpec},
    VultenDataType, VultenInstance,
};

use super::{
    get_block_dims, MatmulPipelineSpec, MatmulPushConst, BROADCAST_A, BROADCAST_B, BROADCAST_NONE,
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
        (a.dims[2], a.dims[1])
    } else {
        (a.dims[1], a.dims[2])
    };
    let mat_b_post: (i64, i64) = if trans_b {
        (b.dims[2], b.dims[1])
    } else {
        (b.dims[1], b.dims[2])
    };
    let block_dims = get_block_dims(mat_a_post, mat_b_post);
    let num_blocks_x = (mat_a_post.0 as f32 / block_dims.0 as f32).ceil() as i64;
    let num_blocks_y = (mat_b_post.1 as f32 / block_dims.1 as f32).ceil() as i64;
    let num_batch = a.dims[0].max(b.dims[0]);
    let broadcast = if a.dims[0] == b.dims[0] {
        BROADCAST_NONE
    } else if a.dims[0] == 1 {
        BROADCAST_A
    } else {
        BROADCAST_B
    };

    let spec = MatmulPipelineSpec {
        local_x: inst.device_props.sub_group_size.max(1),
        block_size_x: block_dims.0,
        block_size_y: block_dims.1,
        bk_cont: mat_a_post.1 as u32 / block_dims.0,
        a_x: mat_a_post.0 as u32,
        a_y: mat_a_post.1 as u32,
        b_x: mat_b_post.0 as u32,
        b_y: mat_b_post.1 as u32,
        inline_trans_a: trans_a,
        inline_trans_b: trans_b,
        bk_num_y: num_blocks_y as u32,
        broadcast,
        d_type,
    };
    let pipeline = inst.get_pipeline_from_spec(PipelineSpecs::Matmul(spec.clone()));

    let descriptors = inst
        .get_descriptor_set(DescriptorType::STORAGE_BUFFER, pipeline.clone())
        .unwrap();

    let a_desc_buff = a.buff.get_descriptor_info()?;
    let b_desc_buff = b.buff.get_descriptor_info()?;
    let output_desc_buff = output.buff.get_descriptor_info()?;

    let write_sets = [
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&a_desc_buff),
        WriteDescriptorSet::default()
            .dst_set(descriptors.descriptor[0])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(DescriptorType::STORAGE_BUFFER)
            .buffer_info(&b_desc_buff),
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

    let mut push = MatmulPushConst {
        start_x: 0,
        stop_x: (num_blocks_x * num_blocks_y) as u32,
        offset: 0,
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

    let chunk_size: i64 = inst.device_props.max_work_group[0] as i64 * spec.local_x as i64;
    if num_blocks_x * num_blocks_y > chunk_size {
        let chunks = (0..num_blocks_x * num_blocks_y).as_chunks(chunk_size);
        for i in 0..num_batch {
            push.offset = i as u32;
            for chunk in chunks.iter() {
                push.start_x = chunk.start as u32;
                push.stop_x = chunk.end as u32;

                let threads = push.stop_x - push.start_x;
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
    } else {
        let threads = ((num_blocks_x * num_blocks_y) as f32 / spec.local_x as f32).ceil() as u32;
        for i in 0..num_batch {
            push.offset = i as u32;
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

    builder.end().build().unwrap();

    let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
    let fence = inst.create_fence().unwrap();

    inst.submit_queue(&q, &[sub_info], fence).unwrap();
    inst.wait_for_fences(&[fence], true).unwrap();

    inst.destroy_fence(fence);
    inst.free_cmd_buffers(&q, cmd_buffs);
    Ok(())
}
