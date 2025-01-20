use std::sync::Arc;

use crate::{
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
    let im2col_output = KernelInput {
        buff: crate::kernels::KernelBuff::Buff(im2col_buff.clone()),
        dims: &im2col_dims,
    };

    im2col::run(
        inst,
        d_type,
        (padd_x as u32, padd_y as u32),
        format,
        (strides.0, strides.1),
        (dilations.0, dilations.1),
        filter_dims,
        input,
        &im2col_output,
    )
    .unwrap();

    let backprop_area = match format {
        ChannelFormat::NHWC => backprop.dims[1] * backprop.dims[2],
        ChannelFormat::NCHW => backprop.dims[2] * backprop.dims[3],
    };
    let in_filter_aera = filter_dims[0] as i64 * filter_dims[1] as i64 * filter_dims[2] as i64;
    let a_dims: Vec<i64> = vec![input.dims[0], backprop_area, in_filter_aera];
    let a = KernelInput {
        buff: crate::kernels::KernelBuff::Buff(im2col_buff),
        dims: &a_dims,
    };
    let b_dims: Vec<i64> = vec![input.dims[0], backprop_area, filter_dims[3] as i64];
    let b = KernelInput {
        buff: backprop.buff.clone(),
        dims: &b_dims,
    };

    let matmul_matrix_dims: Vec<i64> = vec![input.dims[0], in_filter_aera, filter_dims[3] as i64];
    let matmul_buff = Arc::new(inst.create_buffer(
        crate::memory::VultenBufferType::Device,
        matmul_matrix_dims.iter().product::<i64>() as u64 * d_type.size_of().unwrap() as u64,
        false,
        false,
    ));
    let matmul_output = KernelInput {
        buff: crate::kernels::KernelBuff::Buff(matmul_buff),
        dims: &matmul_matrix_dims,
    };
    matmul::matmul_batched::run(inst, d_type, &a, true, &b, false, &matmul_output).unwrap();

    let out_matrix_dims: Vec<i64> = vec![1, in_filter_aera, filter_dims[3] as i64];
    let output_redu = KernelInput {
        buff: output.buff.clone(),
        dims: &out_matrix_dims,
    };
    reduce::reduce::run(
        inst,
        d_type,
        reduce::ReduceOp::Sum,
        vec![0],
        &matmul_output,
        &output_redu,
    )
    .unwrap();

    Ok(())
}
