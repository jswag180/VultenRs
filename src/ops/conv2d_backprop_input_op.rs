use core::{panic, slice};
use std::ffi::{c_char, CStr};

use backend::kernels::{conv2d, matmul, ChannelFormat, KernelInput};
use backend::va::VaAddress;
use backend::GOLBAL_DEVICE_VA;
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_KernelBuilder_HostMemory,
    TF_KernelBuilder_TypeConstraint, TF_NewKernelBuilder, TF_OpKernelConstruction,
    TF_OpKernelConstruction_GetAttrInt32List, TF_OpKernelConstruction_GetAttrString,
    TF_OpKernelContext, TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::log_ops;
use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;

#[derive(Debug, Default)]
#[repr(C)]
struct Conv2DInfo {
    format: ChannelFormat,
    padding: conv2d::Padding,
    strides: [i32; 4],
    dilations: [i32; 4],
}

#[no_mangle]
extern "C" fn create_conv2d_backprop_input(ctx: *mut TF_OpKernelConstruction) -> *mut c_void {
    let mut info = Box::<Conv2DInfo>::default();

    let status = SafeStatus::new();
    let mut c_str: [c_char; 5] = [0; 5];
    unsafe {
        TF_OpKernelConstruction_GetAttrString(
            ctx,
            c"data_format".as_ptr(),
            c_str.as_mut_ptr(),
            5,
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!(
                "TF_OpKernelConstruction_GetAttrString for data_format return status {:?}",
                status.get_code()
            );
            panic!();
        }
        info.format = CStr::from_ptr(c_str.as_ptr())
            .to_string_lossy()
            .as_ref()
            .try_into()
            .unwrap();
    }

    let mut c_str: [c_char; 9] = [0; 9];
    unsafe {
        TF_OpKernelConstruction_GetAttrString(
            ctx,
            c"padding".as_ptr(),
            c_str.as_mut_ptr(),
            9,
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!(
                "TF_OpKernelConstruction_GetAttrString for padding return status {:?}",
                status.get_code()
            );
            panic!();
        }
        info.padding = CStr::from_ptr(c_str.as_ptr())
            .to_string_lossy()
            .as_ref()
            .try_into()
            .unwrap();
    }

    unsafe {
        TF_OpKernelConstruction_GetAttrInt32List(
            ctx,
            c"strides".as_ptr(),
            info.strides.as_mut_ptr(),
            4,
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!(
                "TF_OpKernelConstruction_GetAttrInt32List for strides return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }

    unsafe {
        TF_OpKernelConstruction_GetAttrInt32List(
            ctx,
            c"dilations".as_ptr(),
            info.dilations.as_mut_ptr(),
            4,
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!(
                "TF_OpKernelConstruction_GetAttrInt32List for dilations return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }

    Box::leak(info) as *mut Conv2DInfo as *mut c_void
}

#[no_mangle]
extern "C" fn compute_conv2d_backprop_input(info_ptr: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let info: &Conv2DInfo = unsafe { &*(info_ptr as *const Conv2DInfo) };
    ///////////////////////////////////////////////
    // TODO: Fix
    if info.format == ChannelFormat::NCHW {
        todo!("idk why it does not work");
    }
    ///////////////////////////////////////////////
    if info.padding == conv2d::Padding::Explicit {
        error!("Explicit padding in Conv2D is not supported!");
        return;
    }
    match info.format {
        ChannelFormat::NHWC => {
            if info.strides[0] != 1 || info.strides[3] != 1 {
                error!("Strides in batch or depth not suported in Conv2D!");
                return;
            }
            if info.dilations[0] != 1 || info.dilations[3] != 1 {
                error!("Dilations in batch or depth not suported in Conv2D!");
                return;
            }
        }
        ChannelFormat::NCHW => {
            if info.strides[0] != 1 || info.strides[1] != 1 {
                error!("Strides in batch or depth not suported in Conv2D!");
                return;
            }
            if info.dilations[0] != 1 || info.dilations[1] != 1 {
                error!("Dilations in batch or depth not suported in Conv2D!");
                return;
            }
        }
    }
    let stride_h = match info.format {
        ChannelFormat::NHWC => info.strides[1],
        ChannelFormat::NCHW => info.strides[2],
    };
    let stride_w = match info.format {
        ChannelFormat::NHWC => info.strides[2],
        ChannelFormat::NCHW => info.strides[3],
    };
    let dilation_h = match info.format {
        ChannelFormat::NHWC => info.dilations[1],
        ChannelFormat::NCHW => info.dilations[2],
    };
    let dilation_w = match info.format {
        ChannelFormat::NHWC => info.dilations[2],
        ChannelFormat::NCHW => info.dilations[3],
    };

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };

    let input_tensor = unsafe { SafeTensor::from_input_host(0, ctx, &status) };
    let input_dims: &[i32] = unsafe {
        slice::from_raw_parts(
            input_tensor.get_host_data().unwrap() as *const i32,
            input_tensor.total_elements as usize,
        )
    };
    if input_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            input_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    if input_tensor.dims.len() != 1 {
        error!(
            "Conv2D input needs to be 1 dim got: {:?}",
            input_tensor.dims
        );
        return;
    }
    if input_tensor.total_elements != 4 {
        error!(
            "Conv2D input needs to have 4 elements got: {:?}",
            input_tensor.dims
        );
        return;
    }
    let input_h = match info.format {
        ChannelFormat::NHWC => input_dims[1],
        ChannelFormat::NCHW => input_dims[2],
    };
    let input_w = match info.format {
        ChannelFormat::NHWC => input_dims[2],
        ChannelFormat::NCHW => input_dims[3],
    };
    let input_d = match info.format {
        ChannelFormat::NHWC => input_dims[3],
        ChannelFormat::NCHW => input_dims[1],
    };

    let filter_tensor = unsafe { SafeTensor::from_input_device(1, ctx, &status) };
    if filter_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Filter tensor is to big {:} > {:}",
            filter_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    if filter_tensor.dims.len() != 4 {
        error!(
            "Conv2D filters needs to be 4 dims got: {:?}",
            filter_tensor.dims
        );
        return;
    }
    if filter_tensor.dims[2] != input_d as i64 {
        error!(
            "Input channels {:?} does not match filter in_channels {:?}",
            input_d, filter_tensor.dims[2]
        );
        return;
    }

    let backprop_tensor = unsafe { SafeTensor::from_input_device(2, ctx, &status) };
    if backprop_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Backprop tensor is to big {:} > {:}",
            backprop_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    if backprop_tensor.dims.len() != 4 {
        error!(
            "Conv2D backprop needs to be 4 dims got: {:?}",
            input_tensor.dims
        );
        return;
    }
    let backprop_h = match info.format {
        ChannelFormat::NHWC => backprop_tensor.dims[1],
        ChannelFormat::NCHW => backprop_tensor.dims[2],
    };
    let backprop_w = match info.format {
        ChannelFormat::NHWC => backprop_tensor.dims[2],
        ChannelFormat::NCHW => backprop_tensor.dims[3],
    };
    let backprop_d = match info.format {
        ChannelFormat::NHWC => backprop_tensor.dims[3],
        ChannelFormat::NCHW => backprop_tensor.dims[1],
    };
    let mut padd_x = 0;
    let mut output_x = 0;
    conv2d::get_windowed_ouput(
        input_h.into(),
        filter_tensor.dims[0],
        dilation_h as i64,
        stride_h as i64,
        &info.padding,
        &mut output_x,
        &mut padd_x,
    )
    .unwrap();
    let mut padd_y = 0;
    let mut output_y = 0;
    conv2d::get_windowed_ouput(
        input_w.into(),
        filter_tensor.dims[1],
        dilation_w as i64,
        stride_w as i64,
        &info.padding,
        &mut output_y,
        &mut padd_y,
    )
    .unwrap();
    if backprop_tensor.dims[0] != input_dims[0].into() {
        error!(
            "Conv2D backprop needs to have same batch as input got Input:{:?} Backprop{:?}",
            input_tensor.dims, backprop_tensor.dims
        );
        return;
    }
    if backprop_h != output_x || backprop_w != output_y {
        error!(
            "Conv2D backprop dims incorrect got Input:{:?} Backprop{:?}",
            input_tensor.dims, backprop_tensor.dims
        );
        return;
    }

    let out_dims: Vec<i64> = vec![
        input_dims[0] as i64,
        input_dims[1] as i64,
        input_dims[2] as i64,
        input_dims[3] as i64,
    ];
    let output_tensor =
        unsafe { SafeTensor::new_output(0, out_dims, filter_tensor.d_type, ctx, &status) };

    log_ops!(
        "Running Conv2DBackpropInput\n  Device: {:}\n  Stream: {:p}\n  Input: {:?}\n  Filters: {:?}\n  Backprop: {:?}\n  Format: {:?}\n  Padding: {:?}\n  Strides: {:?}\n  Dilations: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        input_dims,
        filter_tensor,
        backprop_tensor,
        info.format,
        info.padding,
        info.strides,
        info.dilations,
        output_tensor
    );

    if filter_tensor.is_empty {
        return;
    }

    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(filter_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(backprop_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(output_tensor.get_device_data().unwrap())
    );

    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(filter_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(backprop_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(output_tensor.get_device_data().unwrap())
        .is_ok());

    let a_dims: Vec<i64> = vec![backprop_tensor.dims[0], backprop_h * backprop_w, backprop_d];
    let a = KernelInput {
        addr: backprop_tensor.get_device_data().unwrap(),
        dims: &a_dims,
    };
    let b_dims: Vec<i64> = vec![
        1,
        filter_tensor.dims[0] * filter_tensor.dims[1] * filter_tensor.dims[2],
        backprop_d,
    ];
    let b = KernelInput {
        addr: filter_tensor.get_device_data().unwrap(),
        dims: &b_dims,
    };

    let mat_mul_tensor = unsafe {
        SafeTensor::new_temp(
            vec![input_dims[0] as i64 * a_dims[1] * b_dims[1]],
            filter_tensor.d_type,
            ctx,
            &status,
        )
    };
    let mat_mul_out = KernelInput {
        addr: mat_mul_tensor.get_device_data().unwrap(),
        dims: &[a_dims[1], b_dims[1]],
    };
    matmul::matmul_batched::run(
        inst,
        filter_tensor.d_type.into(),
        a,
        false,
        b,
        true,
        mat_mul_out,
    )
    .unwrap();

    let output = KernelInput {
        addr: output_tensor.get_device_data().unwrap(),
        dims: &output_tensor.dims,
    };
    conv2d::col2im::run(
        inst,
        filter_tensor.d_type.into(),
        (padd_x as u32, padd_y as u32),
        info.format,
        (stride_h as u32, stride_w as u32),
        (dilation_h as u32, dilation_w as u32),
        &filter_tensor.dims,
        &backprop_tensor.dims,
        mat_mul_tensor.get_device_data().unwrap(),
        output,
    )
    .unwrap();
}

#[no_mangle]
extern "C" fn destroy_conv2d_backprop_input(info: *mut c_void) {
    let info_box: Box<Conv2DInfo> = unsafe { Box::from_raw(info as *mut Conv2DInfo) };
    drop(info_box);
}

fn register_conv2d_backprop_input_kernel(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let builder = unsafe {
        TF_NewKernelBuilder(
            c"Conv2DBackpropInput".as_ptr(),
            device_type,
            Some(create_conv2d_backprop_input),
            Some(compute_conv2d_backprop_input),
            Some(destroy_conv2d_backprop_input),
        )
    };

    unsafe {
        TF_KernelBuilder_TypeConstraint(builder, c"T".as_ptr(), d_type, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_KernelBuilder_TypeConstraint return status {:?}",
                status.get_code()
            );
            panic!();
        }

        TF_KernelBuilder_HostMemory(builder, c"input_sizes".as_ptr());

        TF_RegisterKernelBuilder(
            c"Conv2DBackpropInput".as_ptr(),
            builder,
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }
}

pub fn register_conv2d_backprop_input_op(device_type: *const c_char) {
    register_conv2d_backprop_input_kernel(device_type, TF_DataType_TF_FLOAT);
}
