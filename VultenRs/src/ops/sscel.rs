use std::ffi::c_char;

use backend::kernels::{binary, reduce, ssxent, unary, KernelInput};
use backend::va::VaAddress;
use backend::GOLBAL_DEVICE_VA;
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_KernelBuilder_TypeConstraint, TF_NewKernelBuilder,
    TF_OpKernelContext, TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

#[no_mangle]
extern "C" fn compute_sscel(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };
    let _prof = profile!(
        "SparseSoftmaxCrossEntropyWithLogits".to_string(),
        inst.dev_num
    );

    let features_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if features_tensor.total_elements > u32::MAX as i64 {
        error!(
            "features tensor is to big {:} > {:}",
            features_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let labels_tensor = unsafe { SafeTensor::from_input_device(1, ctx, &status) };
    if labels_tensor.total_elements > u32::MAX as i64 {
        error!(
            "labels tensor is to big {:} > {:}",
            labels_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let loss = unsafe {
        SafeTensor::new_output(
            0,
            vec![features_tensor.dims[0]],
            features_tensor.d_type,
            ctx,
            &status,
        )
    };
    let backprop =
        unsafe { features_tensor.new_output_like(1, features_tensor.d_type, ctx, &status) };

    log_ops!(
        "Running SparseSoftmaxCrossEntropyWithLogits\n  Device: {:}\n  Stream: {:p}\n  Features: {:?}\n  Labels: {:?}\n  Loss: {:?}\n  Backprop: {:?}",
        inst.dev_num,
        stream,
        features_tensor,
        labels_tensor,
        loss,
        backprop
    );

    if features_tensor.is_empty {
        return;
    }

    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(features_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(labels_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(loss.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(backprop.get_device_data().unwrap())
    );

    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(features_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(labels_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(loss.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(backprop.get_device_data().unwrap())
        .is_ok());

    let max_features = unsafe {
        SafeTensor::new_temp(
            vec![features_tensor.dims[0], 1],
            features_tensor.d_type,
            ctx,
            &status,
        )
    };
    let scratch_exp = unsafe {
        SafeTensor::new_temp(backprop.dims.clone(), features_tensor.d_type, ctx, &status)
    };
    let scratch = unsafe {
        SafeTensor::new_temp(
            vec![features_tensor.dims[0]],
            features_tensor.d_type,
            ctx,
            &status,
        )
    };
    let loss_fat = unsafe {
        SafeTensor::new_temp(backprop.dims.clone(), features_tensor.d_type, ctx, &status)
    };

    let features_input = KernelInput {
        buff: features_tensor.get_device_data().unwrap().into(),
        dims: &features_tensor.dims,
    };
    let max_features_input = KernelInput {
        buff: max_features.get_device_data().unwrap().into(),
        dims: &max_features.dims,
    };
    // maxFeatures
    reduce::reduce::run(
        inst,
        features_tensor.d_type.into(),
        reduce::ReduceOp::Max,
        vec![1],
        &features_input,
        &max_features_input,
    )
    .unwrap();

    // features - maxFeatures
    let backprop_input = KernelInput {
        buff: backprop.get_device_data().unwrap().into(),
        dims: &backprop.dims,
    };
    binary::binary_broad::run(
        inst,
        features_tensor.d_type.into(),
        binary::BinaryOp::Sub,
        &features_input,
        &max_features_input,
        &backprop_input,
    )
    .unwrap();

    // exp
    unary::run(
        inst,
        features_tensor.d_type.into(),
        unary::UnaryOp::Exp,
        &backprop.get_device_data().unwrap().into(),
        &scratch_exp.get_device_data().unwrap().into(),
        backprop.total_elements,
    )
    .unwrap();

    // sum(1)
    let scratch_exp_input = KernelInput {
        buff: scratch_exp.get_device_data().unwrap().into(),
        dims: &scratch_exp.dims,
    };
    let scratch_input = KernelInput {
        buff: scratch.get_device_data().unwrap().into(),
        dims: &scratch.dims,
    };
    reduce::reduce::run(
        inst,
        features_tensor.d_type.into(),
        reduce::ReduceOp::Sum,
        vec![1],
        &scratch_exp_input,
        &scratch_input,
    )
    .unwrap();

    let labels_input = KernelInput {
        buff: labels_tensor.get_device_data().unwrap().into(),
        dims: &labels_tensor.dims,
    };
    let loss_fat_input = KernelInput {
        buff: loss_fat.get_device_data().unwrap().into(),
        dims: &loss_fat.dims,
    };
    ssxent::run(
        inst,
        features_tensor.d_type.into(),
        labels_tensor.d_type.into(),
        &scratch_input,
        &backprop_input,
        &labels_input,
        &loss_fat_input,
        &backprop_input,
    )
    .unwrap();

    let loss_input = KernelInput {
        buff: loss.get_device_data().unwrap().into(),
        dims: &loss.dims,
    };
    reduce::reduce::run(
        inst,
        features_tensor.d_type.into(),
        reduce::ReduceOp::Sum,
        vec![1],
        &loss_fat_input,
        &loss_input,
    )
    .unwrap();
}

fn register_sscel_kernel(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let builder = unsafe {
        TF_NewKernelBuilder(
            c"SparseSoftmaxCrossEntropyWithLogits".as_ptr(),
            device_type,
            None,
            Some(compute_sscel),
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

        TF_RegisterKernelBuilder(c"SparseSoftmaxCrossEntropyWithLogits".as_ptr(), builder, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }
}

pub fn register_sscel_op(device_type: *const c_char) {
    register_sscel_kernel(device_type, TF_DataType_TF_FLOAT);
}
