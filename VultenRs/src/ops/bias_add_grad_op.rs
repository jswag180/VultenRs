use std::ffi::{c_char, CStr};

use backend::kernels::{reduce, ChannelFormat, KernelInput};
use backend::va::VaAddress;
use backend::GOLBAL_DEVICE_VA;
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32, TF_DataType_TF_INT64,
    TF_DataType_TF_UINT32, TF_DataType_TF_UINT64, TF_KernelBuilder_TypeConstraint,
    TF_NewKernelBuilder, TF_OpKernelConstruction, TF_OpKernelConstruction_GetAttrString,
    TF_OpKernelContext, TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

#[derive(Debug, Default)]
#[repr(C)]
struct BiasAddGradInfo {
    format: ChannelFormat,
}

#[no_mangle]
extern "C" fn create_bias_add_grad(ctx: *mut TF_OpKernelConstruction) -> *mut c_void {
    let mut info = Box::<BiasAddGradInfo>::default();

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
                "TF_OpKernelConstruction_GetAttrBool for keep_dims return status {:?}",
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

    Box::leak(info) as *mut BiasAddGradInfo as *mut c_void
}

#[no_mangle]
extern "C" fn compute_bias_add_grad(info_ptr: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let info: &BiasAddGradInfo = unsafe { &*(info_ptr as *const BiasAddGradInfo) };

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };
    let _prof = profile!("BiasAddGrad".to_string(), inst.dev_num);

    let input_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if input_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            input_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let mut axis_vec: Vec<u32> = Vec::with_capacity(input_tensor.dims.len() - 1);
    let mut output_dims: Vec<i64> = vec![1];
    match info.format {
        ChannelFormat::NHWC => {
            output_dims[0] = *input_tensor.dims.last().unwrap();
            for i in 0..(input_tensor.dims.len() - 1) {
                axis_vec.push(i as u32);
            }
        }
        ChannelFormat::NCHW => {
            output_dims[0] = input_tensor.dims[1];
            for i in 0..input_tensor.dims.len() {
                if i != 1 {
                    axis_vec.push(i as u32);
                }
            }
        }
    }
    axis_vec.reverse();

    let output_tensor =
        unsafe { SafeTensor::new_output(0, output_dims, input_tensor.d_type, ctx, &status) };

    log_ops!(
        "Running BiasAddGrad\n  Device: {:}\n  Stream: {:p}\n  Format: {:?}\n  Input: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        info.format,
        input_tensor,
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
        VaAddress::get_device_num(output_tensor.get_device_data().unwrap())
    );

    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(input_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(output_tensor.get_device_data().unwrap())
        .is_ok());

    let input = KernelInput {
        addr: input_tensor.get_device_data().unwrap(),
        dims: &input_tensor.dims,
    };
    let output = KernelInput {
        addr: output_tensor.get_device_data().unwrap(),
        dims: &output_tensor.dims,
    };

    reduce::reduce::run(
        inst,
        input_tensor.d_type.into(),
        reduce::ReduceOp::Sum,
        axis_vec,
        input,
        output,
    )
    .unwrap();
}

#[no_mangle]
extern "C" fn destroy_bias_add_grad(info: *mut c_void) {
    let info_box: Box<BiasAddGradInfo> = unsafe { Box::from_raw(info as *mut BiasAddGradInfo) };
    drop(info_box);
}

fn register_bias_add_grad_kernel(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let builder = unsafe {
        TF_NewKernelBuilder(
            c"BiasAddGrad".as_ptr(),
            device_type,
            Some(create_bias_add_grad),
            Some(compute_bias_add_grad),
            Some(destroy_bias_add_grad),
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

        TF_RegisterKernelBuilder(c"BiasAddGrad".as_ptr(), builder, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }
}

pub fn register_bias_add_grad_op(device_type: *const c_char) {
    register_bias_add_grad_kernel(device_type, TF_DataType_TF_FLOAT);
    register_bias_add_grad_kernel(device_type, TF_DataType_TF_INT32);
    register_bias_add_grad_kernel(device_type, TF_DataType_TF_UINT32);
    register_bias_add_grad_kernel(device_type, TF_DataType_TF_INT64);
    register_bias_add_grad_kernel(device_type, TF_DataType_TF_UINT64);
}
