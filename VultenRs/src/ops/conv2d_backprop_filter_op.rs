use core::slice;
use std::ffi::{c_char, CStr};

use backend::kernels::{conv2d, ChannelFormat, KernelInput};
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

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

#[derive(Debug, Default)]
#[repr(C)]
struct Conv2DInfo {
    format: ChannelFormat,
    padding: conv2d::Padding,
    strides: [i32; 4],
    dilations: [i32; 4],
}

#[no_mangle]
extern "C" fn create_conv2d_backprop_filter(ctx: *mut TF_OpKernelConstruction) -> *mut c_void {
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
extern "C" fn compute_conv2d_backprop_filter(info_ptr: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let info: &Conv2DInfo = unsafe { &*(info_ptr as *const Conv2DInfo) };
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
    let _prof = profile!("Conv2DBackpropFilter".to_string(), inst.dev_num);

    let input_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if input_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            input_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    if input_tensor.dims.len() != 4 {
        error!(
            "Conv2D input needs to be 4 dims got: {:?}",
            input_tensor.dims
        );
        return;
    }
    let input_d = match info.format {
        ChannelFormat::NHWC => input_tensor.dims[3],
        ChannelFormat::NCHW => input_tensor.dims[1],
    };

    let filters_tensor = unsafe { SafeTensor::from_input_host(1, ctx, &status) };
    let filter_dims: &[i32] = unsafe {
        slice::from_raw_parts(
            filters_tensor.get_host_data().unwrap() as *const i32,
            filters_tensor.total_elements as usize,
        )
    };
    if filters_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Filters tensor is to big {:} > {:}",
            filters_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    if filters_tensor.dims.len() != 1 {
        error!(
            "Conv2D filters needs to be 1 dim got: {:?}",
            filters_tensor.dims
        );
        return;
    }
    if filters_tensor.total_elements != 4 {
        error!(
            "Conv2D filters needs to have 4 elements got: {:?}",
            filters_tensor.dims
        );
        return;
    }
    if filter_dims[2] != input_d as i32 {
        error!(
            "Input channels {:?} does not match filter in_channels {:?}",
            input_d, filter_dims[2]
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

    let out_dims: Vec<i64> = vec![
        filter_dims[0] as i64,
        filter_dims[1] as i64,
        filter_dims[2] as i64,
        filter_dims[3] as i64,
    ];
    let output_tensor =
        unsafe { SafeTensor::new_output(0, out_dims, input_tensor.d_type, ctx, &status) };

    log_ops!(
        "Running Conv2DBackpropFilter\n  Device: {:}\n  Stream: {:p}\n  Input: {:?}\n  Filters: {:?}\n  Backprop: {:?}\n  Format: {:?}\n  Padding: {:?}\n  Strides: {:?}\n  Dilations: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        input_tensor,
        filter_dims,
        backprop_tensor,
        info.format,
        info.padding,
        info.strides,
        info.dilations,
        output_tensor
    );

    if input_tensor.is_empty {
        return;
    }

    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(input_tensor.get_device_data().unwrap())
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
        .find_va(input_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(backprop_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(output_tensor.get_device_data().unwrap())
        .is_ok());

    let input = KernelInput {
        buff: input_tensor.get_device_data().unwrap().into(),
        dims: &input_tensor.dims,
    };
    let backprop = KernelInput {
        buff: backprop_tensor.get_device_data().unwrap().into(),
        dims: &backprop_tensor.dims,
    };
    let output = KernelInput {
        buff: output_tensor.get_device_data().unwrap().into(),
        dims: &output_tensor.dims,
    };
    conv2d::backprop_filter::run(
        inst,
        input_tensor.d_type.into(),
        &info.padding,
        info.format,
        (stride_h as u32, stride_w as u32),
        (dilation_h as u32, dilation_w as u32),
        filter_dims,
        &input,
        &backprop,
        &output,
    )
    .unwrap();
}

#[no_mangle]
extern "C" fn destroy_conv2d_backprop_filter(info: *mut c_void) {
    let info_box: Box<Conv2DInfo> = unsafe { Box::from_raw(info as *mut Conv2DInfo) };
    drop(info_box);
}

fn register_conv2d_backprop_filter_kernel(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let builder = unsafe {
        TF_NewKernelBuilder(
            c"Conv2DBackpropFilter".as_ptr(),
            device_type,
            Some(create_conv2d_backprop_filter),
            Some(compute_conv2d_backprop_filter),
            Some(destroy_conv2d_backprop_filter),
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

        TF_KernelBuilder_HostMemory(builder, c"filter_sizes".as_ptr());

        TF_RegisterKernelBuilder(
            c"Conv2DBackpropFilter".as_ptr(),
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

pub fn register_conv2d_backprop_filter_op(device_type: *const c_char) {
    register_conv2d_backprop_filter_kernel(device_type, TF_DataType_TF_FLOAT);
}
