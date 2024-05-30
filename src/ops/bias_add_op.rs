use std::ffi::c_char;

use backend::kernels::binary;
use backend::va::VaAddress;
use backend::GOLBAL_DEVICE_VA;
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32, TF_DataType_TF_INT64,
    TF_KernelBuilder_TypeConstraint, TF_NewKernelBuilder, TF_OpKernelContext,
    TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::log_ops;
use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;

#[no_mangle]
extern "C" fn compute_bias_add(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };

    let input_tensor = unsafe { SafeTensor::from_input(0, ctx, &status) };
    if input_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            input_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let bias_tensor = unsafe { SafeTensor::from_input(1, ctx, &status) };
    if bias_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Bias tensor is to big {:} > {:}",
            bias_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let output_tensor =
        unsafe { input_tensor.new_output_like(0, input_tensor.d_type, ctx, &status) };

    log_ops!(
        "Running BiasAdd\n  Device: {:}\n  Stream: {:p}\n  Input: {:?}\n  Bias: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        input_tensor,
        bias_tensor,
        output_tensor
    );

    if input_tensor.is_empty || bias_tensor.is_empty {
        return;
    }

    debug_assert_eq!(inst.dev_num, VaAddress::get_device_num(input_tensor.data));
    debug_assert_eq!(inst.dev_num, VaAddress::get_device_num(bias_tensor.data));
    debug_assert_eq!(inst.dev_num, VaAddress::get_device_num(output_tensor.data));

    unsafe {
        debug_assert!(GOLBAL_DEVICE_VA.find_va(input_tensor.data).is_ok());
        debug_assert!(GOLBAL_DEVICE_VA.find_va(bias_tensor.data).is_ok());
        debug_assert!(GOLBAL_DEVICE_VA.find_va(output_tensor.data).is_ok());
    }

    binary::binary_simple::run(
        inst,
        input_tensor.d_type.into(),
        binary::BinaryOp::Add,
        input_tensor.data,
        input_tensor.total_elements,
        bias_tensor.data,
        bias_tensor.total_elements,
        output_tensor.data,
    )
    .unwrap();
}

fn register_bias_add_kernel(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let builder = unsafe {
        TF_NewKernelBuilder(
            c"BiasAdd".as_ptr(),
            device_type,
            None,
            Some(compute_bias_add),
            None,
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

        TF_RegisterKernelBuilder(c"BiasAdd".as_ptr(), builder, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }
}

pub fn register_bias_add_op(device_type: *const c_char) {
    register_bias_add_kernel(device_type, TF_DataType_TF_FLOAT);
    register_bias_add_kernel(device_type, TF_DataType_TF_INT32);
    register_bias_add_kernel(device_type, TF_DataType_TF_INT64);
}
