use std::sync::Arc;

use ash::vk::{
    AccessFlags, DependencyFlags, MemoryBarrier, PipelineStageFlags, QueueFlags, SubmitInfo,
};

use crate::{
    cmd_buff::CommandBufferBuilder,
    kernels::{
        matmul::{self},
        ChannelFormat, KernelBuff, KernelInput,
    },
    VultenDataType, VultenInstance,
};

use super::{col2im, get_windowed_ouput, Padding};

pub fn run(
    inst: &VultenInstance,
    d_type: VultenDataType,
    padding: &Padding,
    format: ChannelFormat,
    strides: (u32, u32),
    dilations: (u32, u32),
    input_dims: &[i32],
    filters: &KernelInput,
    backprop: &KernelInput,
    output: &KernelInput,
) -> Result<(), &'static str> {
    let backprop_h = match format {
        ChannelFormat::NHWC => backprop.dims[1],
        ChannelFormat::NCHW => backprop.dims[2],
    };
    let backprop_w = match format {
        ChannelFormat::NHWC => backprop.dims[2],
        ChannelFormat::NCHW => backprop.dims[3],
    };
    let backprop_d = match format {
        ChannelFormat::NHWC => backprop.dims[3],
        ChannelFormat::NCHW => backprop.dims[1],
    };
    let input_h = match format {
        ChannelFormat::NHWC => input_dims[1],
        ChannelFormat::NCHW => input_dims[2],
    };
    let input_w = match format {
        ChannelFormat::NHWC => input_dims[2],
        ChannelFormat::NCHW => input_dims[3],
    };
    let mut padd_x = 0;
    let mut output_x = 0;
    get_windowed_ouput(
        input_h.into(),
        filters.dims[0],
        dilations.0 as i64,
        strides.0 as i64,
        padding,
        &mut output_x,
        &mut padd_x,
    )
    .unwrap();
    let mut padd_y = 0;
    let mut output_y = 0;
    get_windowed_ouput(
        input_w.into(),
        filters.dims[1],
        dilations.1 as i64,
        strides.1 as i64,
        padding,
        &mut output_y,
        &mut padd_y,
    )
    .unwrap();

    //MatMul
    let a_dims: Vec<i64> = vec![backprop.dims[0], backprop_h * backprop_w, backprop_d];
    let b_dims: Vec<i64> = vec![
        1,
        filters.dims[0] * filters.dims[1] * filters.dims[2],
        backprop_d,
    ];
    let mat_mul_buff = Arc::new(inst.create_buffer(
        crate::memory::VultenBufferType::Device,
        (input_dims[0] as i64 * a_dims[1] * b_dims[1]) as u64 * d_type.size_of().unwrap() as u64,
        false,
        false,
    ));
    let mat_mul_out = KernelBuff::Buff(mat_mul_buff.clone());
    let mat_mul_dims = [a_dims[0].max(b_dims[0]), a_dims[1], b_dims[1]];

    let mut matmul = matmul::MatMulKernel::new(inst, d_type)
        .a(backprop.buff.clone(), &a_dims, false)?
        .b(filters.buff.clone(), &b_dims, true)?
        .output(mat_mul_out, &mat_mul_dims)?
        .build(None)?;
    let matmul_pipeline = matmul.get_pipeline()?;
    let matmul_descriptors = matmul.get_descriptors(matmul_pipeline.clone())?;

    //Col2Img
    let mut col2img = col2im::Col2ImgKernel::new(
        inst,
        d_type,
        (padd_x as u32, padd_y as u32),
        strides,
        dilations,
        format,
    )
    .filter(filters.dims)?
    .backprop(backprop.dims)?
    .input(KernelBuff::Buff(mat_mul_buff))?
    .output(output.buff.clone(), output.dims)?;
    let col2img_pipeline = col2img.get_pipeline()?;
    let col2img_descriptors = col2img.get_descriptors(col2img_pipeline.clone())?;

    //Record
    let q = inst.get_queue(QueueFlags::COMPUTE);
    let cmd_buffs = inst
        .create_cmd_buffers(1, &q)
        .or(Err("Could not create command buffers"))?;
    let mut builder = CommandBufferBuilder::new(cmd_buffs[0], &inst.device).begin();
    let barrier = MemoryBarrier::default()
        .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

    //MatMul
    builder = matmul
        .record(builder, matmul_pipeline, &matmul_descriptors)?
        .pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );

    //Col2Img
    col2img
        .record(builder, col2img_pipeline, &col2img_descriptors)?
        .end()
        .build()?;

    //Submit
    let sub_info = SubmitInfo::default().command_buffers(&cmd_buffs);
    let fence = inst.create_fence().or(Err("Could not create fence"))?;

    inst.submit_queue(&q, &[sub_info], fence)
        .or(Err("Could not submit queue"))?;
    inst.wait_for_fences(&[fence], true)
        .or(Err("Fence timed out"))?;

    inst.destroy_fence(fence);
    inst.free_cmd_buffers(&q, cmd_buffs);

    Ok(())
}
