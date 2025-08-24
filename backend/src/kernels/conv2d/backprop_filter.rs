use std::sync::Arc;

use ash::vk::{
    AccessFlags, DependencyFlags, MemoryBarrier, PipelineStageFlags, QueueFlags, SubmitInfo,
};

use crate::{
    cmd_buff::CommandBufferBuilder,
    kernels::{matmul, reduce, ChannelFormat, KernelInput},
    VultenDataType, VultenInstance,
};

use super::{get_windowed_ouput, im2col, Padding};

pub fn run(
    inst: &VultenInstance,
    d_type: VultenDataType,
    padding: &Padding,
    format: ChannelFormat,
    strides: (u32, u32),
    dilations: (u32, u32),
    filter_dims: &[i32],
    input: &KernelInput,
    backprop: &KernelInput,
    output: &KernelInput,
) -> Result<(), &'static str> {
    let input_h = match format {
        ChannelFormat::NHWC => input.dims[1],
        ChannelFormat::NCHW => input.dims[2],
    };
    let input_w = match format {
        ChannelFormat::NHWC => input.dims[2],
        ChannelFormat::NCHW => input.dims[3],
    };

    let mut padd_x = 0;
    let mut output_x = 0;
    get_windowed_ouput(
        input_h,
        filter_dims[0] as i64,
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
        input_w,
        filter_dims[1] as i64,
        dilations.1 as i64,
        strides.1 as i64,
        padding,
        &mut output_y,
        &mut padd_y,
    )
    .unwrap();

    // Im2Col
    let im2col_dims: Vec<i64> = match format {
        ChannelFormat::NHWC => vec![input.dims[0], output_x, output_y, filter_dims[2] as i64],
        ChannelFormat::NCHW => vec![input.dims[0], filter_dims[2] as i64, output_x, output_y],
    };
    let im2col_buff = Arc::new(inst.create_buffer(
        crate::memory::VultenBufferType::Device,
        (im2col_dims.iter().product::<i64>() * filter_dims[0] as i64 * filter_dims[1] as i64)
            as u64
            * d_type.size_of().unwrap() as u64,
        false,
        false,
    ));

    let mut im2col = im2col::Img2ColKernel::new(
        inst,
        d_type,
        (padd_x as u32, padd_y as u32),
        (strides.0, strides.1),
        (dilations.0, dilations.1),
        format,
    )
    .filter(crate::dims::Dims::Slice(filter_dims))?
    .input(input.buff.clone(), crate::dims::Dims::Slice(input.dims))?
    .output(
        crate::kernels::KernelBuff::Buff(im2col_buff.clone()),
        crate::dims::Dims::Slice(&im2col_dims),
    )?;
    let im2col_pipeline = im2col.get_pipeline()?;
    let im2col_desc = im2col.get_descriptors(im2col_pipeline.clone())?;

    // MatMul
    let backprop_area = match format {
        ChannelFormat::NHWC => backprop.dims[1] * backprop.dims[2],
        ChannelFormat::NCHW => backprop.dims[2] * backprop.dims[3],
    };
    let in_filter_aera = filter_dims[0] as i64 * filter_dims[1] as i64 * filter_dims[2] as i64;
    let a_dims: Vec<i64> = vec![input.dims[0], backprop_area, in_filter_aera];
    let b_dims: Vec<i64> = vec![input.dims[0], backprop_area, filter_dims[3] as i64];

    let matmul_matrix_dims: Vec<i64> = vec![input.dims[0], in_filter_aera, filter_dims[3] as i64];
    let matmul_buff = Arc::new(inst.create_buffer(
        crate::memory::VultenBufferType::Device,
        matmul_matrix_dims.iter().product::<i64>() as u64 * d_type.size_of().unwrap() as u64,
        false,
        false,
    ));

    let mut matmul = matmul::MatMulKernel::new(inst, d_type)
        .a(
            crate::kernels::KernelBuff::Buff(im2col_buff.clone()),
            &a_dims,
            true,
        )?
        .b(backprop.buff.clone(), &b_dims, false)?
        .output(
            crate::kernels::KernelBuff::Buff(matmul_buff.clone()),
            &matmul_matrix_dims,
        )?
        .build(None)?;
    let matmul_pipeline = matmul.get_pipeline()?;
    let matmul_desc = matmul.get_descriptors(matmul_pipeline.clone())?;

    //Reduce
    let out_matrix_dims: Vec<i64> = vec![1, in_filter_aera, filter_dims[3] as i64];

    let mut reduce = reduce::ReduceKernel::new(inst, d_type, reduce::ReduceOp::Sum)
        .reduce_dims(vec![0])?
        .input(
            crate::kernels::KernelBuff::Buff(matmul_buff),
            &matmul_matrix_dims,
        )?
        .output(output.buff.clone(), &out_matrix_dims)?
        .build(None)?;
    let reduce_pipeline = reduce.get_pipeline()?;
    let reduce_desc = reduce.get_descriptors(&reduce_pipeline)?;

    //record
    let q = inst.get_queue(QueueFlags::COMPUTE);
    let cmd_buffs = inst
        .create_cmd_buffers(1, &q)
        .or(Err("Could not create command buffers"))?;
    let mut builder = CommandBufferBuilder::new(cmd_buffs[0], &inst.device).begin();
    let barrier = MemoryBarrier::default()
        .src_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)
        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

    //Im2Col
    builder = im2col
        .record(builder, im2col_pipeline, &im2col_desc)?
        .pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );

    //MatMul
    builder = matmul
        .record(builder, matmul_pipeline, &matmul_desc)?
        .pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );

    //Reduce
    reduce
        .record(builder, &reduce_pipeline, &reduce_desc)?
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
