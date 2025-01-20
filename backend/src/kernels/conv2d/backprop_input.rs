use std::sync::Arc;

use crate::{
    kernels::{matmul, ChannelFormat, KernelInput},
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

    let a_dims: Vec<i64> = vec![backprop.dims[0], backprop_h * backprop_w, backprop_d];
    let a = KernelInput {
        buff: backprop.buff.clone(),
        dims: &a_dims,
    };
    let b_dims: Vec<i64> = vec![
        1,
        filters.dims[0] * filters.dims[1] * filters.dims[2],
        backprop_d,
    ];
    let b = KernelInput {
        buff: filters.buff.clone(),
        dims: &b_dims,
    };

    let mat_mul_buff = Arc::new(inst.create_buffer(
        crate::memory::VultenBufferType::Device,
        (input_dims[0] as i64 * a_dims[1] * b_dims[1]) as u64 * d_type.size_of().unwrap() as u64,
        false,
        false,
    ));
    let mat_mul_out = KernelInput {
        buff: crate::kernels::KernelBuff::Buff(mat_mul_buff.clone()),
        dims: &[a_dims[1], b_dims[1]],
    };
    matmul::matmul_batched::run(inst, d_type, &a, false, &b, true, &mat_mul_out).unwrap();

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
    col2im::run(
        inst,
        d_type,
        (padd_x as u32, padd_y as u32),
        format,
        (strides.0, strides.1),
        (dilations.0, dilations.1),
        filters.dims,
        backprop.dims,
        &crate::kernels::KernelBuff::Buff(mat_mul_buff),
        output,
    )
    .unwrap();

    Ok(())
}
