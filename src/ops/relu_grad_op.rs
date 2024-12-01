use std::ffi::c_char;

use backend::kernels::relu;
use backend::va::VaAddress;
use backend::GOLBAL_DEVICE_VA;
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32, TF_DataType_TF_INT64,
    TF_KernelBuilder_TypeConstraint, TF_NewKernelBuilder, TF_OpKernelContext,
    TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

#[no_mangle]
extern "C" fn compute_relu_grad(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };
    let _prof = profile!("ReluGrad".to_string(), inst.dev_num);

    let gradients_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if gradients_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Gradients tensor is to big {:} > {:}",
            gradients_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    let features_tensor = unsafe { SafeTensor::from_input_device(1, ctx, &status) };
    if features_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Features tensor is to big {:} > {:}",
            features_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    if gradients_tensor.is_empty || features_tensor.is_empty {
        let _ =
            unsafe { gradients_tensor.new_output_like(0, gradients_tensor.d_type, ctx, &status) };
        return;
    }
    if gradients_tensor.total_elements != features_tensor.total_elements {
        error!("gradients != features");
        return;
    }

    let output_tensor =
        unsafe { gradients_tensor.new_output_like(0, gradients_tensor.d_type, ctx, &status) };

    log_ops!(
        "Running ReluGrad\n  Device: {:}\n  Stream: {:p}\n  Gradients: {:?}\n  Features: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        gradients_tensor,
        features_tensor,
        output_tensor
    );

    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(gradients_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(features_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(output_tensor.get_device_data().unwrap())
    );

    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(gradients_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(features_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(output_tensor.get_device_data().unwrap())
        .is_ok());

    relu::relu_grad::run(
        inst,
        gradients_tensor.d_type.into(),
        gradients_tensor.get_device_data().unwrap(),
        features_tensor.get_device_data().unwrap(),
        output_tensor.get_device_data().unwrap(),
        gradients_tensor.total_elements,
    )
    .unwrap();
}

fn register_relu_grad_kernel(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let builder = unsafe {
        TF_NewKernelBuilder(
            c"ReluGrad".as_ptr(),
            device_type,
            None,
            Some(compute_relu_grad),
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

        TF_RegisterKernelBuilder(c"ReluGrad".as_ptr(), builder, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }
}

pub fn register_relu_grad_op(device_type: *const c_char) {
    register_relu_grad_kernel(device_type, TF_DataType_TF_FLOAT);
    register_relu_grad_kernel(device_type, TF_DataType_TF_INT32);
    register_relu_grad_kernel(device_type, TF_DataType_TF_INT64);
}
