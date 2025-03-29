use std::ffi::c_char;

use backend::kernels::binary::shape_helper::BroadcastShapeHelper;
use backend::kernels::{binary, reduce, ssxent, unary};
use backend::va::VaAddress;
use backend::{ENV_SETTINGS, GOLBAL_DEVICE_VA};
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

    // maxFeatures
    reduce::ReduceKernel::new(inst, features_tensor.d_type.into(), reduce::ReduceOp::Max)
        .reduce_dims(vec![1])
        .unwrap()
        .input(
            features_tensor.get_device_data().unwrap().into(),
            &features_tensor.dims,
        )
        .unwrap()
        .output(
            max_features.get_device_data().unwrap().into(),
            &max_features.dims,
        )
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    // features - maxFeatures
    let shape_helper =
        BroadcastShapeHelper::new(features_tensor.dims.clone(), max_features.dims.clone()).unwrap();
    binary::BinaryKernel::new(
        inst,
        features_tensor.d_type.into(),
        binary::BinaryOp::Sub,
        shape_helper,
    )
    .a(features_tensor.get_device_data().unwrap().into())
    .unwrap()
    .b(max_features.get_device_data().unwrap().into())
    .unwrap()
    .output(backprop.get_device_data().unwrap().into())
    .unwrap()
    .build(None)
    .unwrap()
    .run()
    .unwrap();

    // exp
    unary::UnaryKernel::new(inst, features_tensor.d_type.into(), unary::UnaryOp::Exp)
        .input(
            backprop.get_device_data().unwrap().into(),
            backprop.total_elements,
        )
        .unwrap()
        .output(scratch_exp.get_device_data().unwrap().into())
        .unwrap()
        .run()
        .unwrap();

    // sum(1)
    reduce::ReduceKernel::new(inst, features_tensor.d_type.into(), reduce::ReduceOp::Sum)
        .reduce_dims(vec![1])
        .unwrap()
        .input(
            scratch_exp.get_device_data().unwrap().into(),
            &scratch_exp.dims,
        )
        .unwrap()
        .output(scratch.get_device_data().unwrap().into(), &scratch.dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
        .unwrap();

    ssxent::SSXENTKernel::new(
        inst,
        features_tensor.d_type.into(),
        labels_tensor.d_type.into(),
    )
    .scratch(scratch.get_device_data().unwrap().into())
    .unwrap()
    .backprop(backprop.get_device_data().unwrap().into(), &backprop.dims)
    .unwrap()
    .labels(labels_tensor.get_device_data().unwrap().into())
    .unwrap()
    .loss_fat(loss_fat.get_device_data().unwrap().into())
    .unwrap()
    .grad(backprop.get_device_data().unwrap().into())
    .unwrap()
    .run()
    .unwrap();

    reduce::ReduceKernel::new(inst, features_tensor.d_type.into(), reduce::ReduceOp::Sum)
        .reduce_dims(vec![1])
        .unwrap()
        .input(loss_fat.get_device_data().unwrap().into(), &loss_fat.dims)
        .unwrap()
        .output(loss.get_device_data().unwrap().into(), &loss.dims)
        .unwrap()
        .build(None)
        .unwrap()
        .run()
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

        TF_RegisterKernelBuilder(
            c"SparseSoftmaxCrossEntropyWithLogits".as_ptr(),
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

pub fn register_sscel_op(device_type: *const c_char) {
    //The lables input can be an int64
    //would be nice to add support for casting to int32 when int64 is disabled
    if !ENV_SETTINGS.disable_int64 {
        register_sscel_kernel(device_type, TF_DataType_TF_FLOAT);
    }
}
