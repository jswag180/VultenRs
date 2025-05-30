use std::ffi::c_char;

use backend::kernels::binary::shape_helper::BroadcastShapeHelper;
use backend::kernels::binary::{self, BinaryOp};
use backend::va::VaAddress;
use backend::{ENV_SETTINGS, GOLBAL_DEVICE_VA};
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_DOUBLE, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32,
    TF_DataType_TF_INT64, TF_DataType_TF_UINT32, TF_DataType_TF_UINT64,
    TF_KernelBuilder_TypeConstraint, TF_NewKernelBuilder, TF_OpKernelContext,
    TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

//#[no_mangle]
extern "C" fn compute_binary<const T: u32>(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };
    let _prof = profile!(
        format!("{:?}", <u32 as TryInto<BinaryOp>>::try_into(T).unwrap()),
        inst.dev_num
    );

    let x_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if x_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            x_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let y_tensor = unsafe { SafeTensor::from_input_device(1, ctx, &status) };
    if y_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            y_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let shape_helper =
        BroadcastShapeHelper::new(x_tensor.dims.clone(), y_tensor.dims.clone()).unwrap();

    log_ops!(
        "Running Binary\n  Device: {:}\n  Stream: {:p}\n  Type: {:?}\n  Op: {:?}\n  X: {:?}\n  Y: {:?}\n  Helper: {:?}",
        inst.dev_num,
        stream,
        x_tensor.d_type,
        <u32 as TryInto<BinaryOp>>::try_into(T).unwrap(),
        x_tensor,
        y_tensor,
        shape_helper
    );

    let output_tensor = unsafe {
        SafeTensor::new_output(
            0,
            shape_helper.out_shape.clone(),
            x_tensor.d_type,
            ctx,
            &status,
        )
    };

    if x_tensor.is_empty || y_tensor.is_empty {
        return;
    }

    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(x_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(y_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(output_tensor.get_device_data().unwrap())
    );

    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(x_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(y_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(output_tensor.get_device_data().unwrap())
        .is_ok());

    binary::BinaryKernel::new(
        inst,
        x_tensor.d_type.into(),
        <u32 as TryInto<BinaryOp>>::try_into(T).unwrap(),
        shape_helper,
    )
    .a(x_tensor.get_device_data().unwrap().into())
    .unwrap()
    .b(y_tensor.get_device_data().unwrap().into())
    .unwrap()
    .output(output_tensor.get_device_data().unwrap().into())
    .unwrap()
    .build(None)
    .unwrap()
    .run()
    .unwrap();
}

fn register_binary_kernel<const T: u32>(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let op_str = match T.try_into().unwrap() {
        BinaryOp::Mul => c"Mul",
        BinaryOp::Add => c"Add",
        BinaryOp::Sub => c"Sub",
        BinaryOp::Div => c"Div",
        BinaryOp::DivNoNan => c"DivNoNan",
        BinaryOp::DivReal => c"RealDiv",
        BinaryOp::Max => c"Maximum",
        BinaryOp::Min => c"Minimum",
        BinaryOp::Pow => c"Pow",
        BinaryOp::SqrDrff => c"SquaredDifference",
        BinaryOp::TanhGrad => c"TanhGrad",
        BinaryOp::ReluGrad => c"ReluGrad",
        BinaryOp::RsqrtGrad => c"RsqrtGrad",
    };

    let builder = unsafe {
        TF_NewKernelBuilder(
            op_str.as_ptr(),
            device_type,
            None,
            Some(compute_binary::<T>),
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

        TF_RegisterKernelBuilder(op_str.as_ptr(), builder, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }

    if <u32 as TryInto<BinaryOp>>::try_into(T).unwrap() == BinaryOp::Add {
        let builder = unsafe {
            TF_NewKernelBuilder(
                c"AddV2".as_ptr(),
                device_type,
                None,
                Some(compute_binary::<T>),
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

            TF_RegisterKernelBuilder(c"AddV2".as_ptr(), builder, status.status_ptr());
            if !status.is_ok() {
                error!(
                    "TF_RegisterKernelBuilder return status {:?}",
                    status.get_code()
                );
                panic!();
            }
        }
    }
}

#[inline(always)]
fn register_type(device_type: *const c_char, d_type: TF_DataType) {
    register_binary_kernel::<{ BinaryOp::Mul as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Add as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Sub as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Div as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Div as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::DivNoNan as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::DivReal as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Max as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Min as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Pow as u32 }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::SqrDrff as u32 }>(device_type, d_type);
}

pub fn register_binary_ops(device_type: *const c_char) {
    register_type(device_type, TF_DataType_TF_FLOAT);
    register_type(device_type, TF_DataType_TF_INT32);
    register_type(device_type, TF_DataType_TF_UINT32);
    if !ENV_SETTINGS.disable_int64 {
        register_type(device_type, TF_DataType_TF_INT64);
        register_type(device_type, TF_DataType_TF_UINT64);
    }

    register_binary_kernel::<{ BinaryOp::TanhGrad as u32 }>(device_type, TF_DataType_TF_FLOAT);

    register_binary_kernel::<{ BinaryOp::ReluGrad as u32 }>(device_type, TF_DataType_TF_FLOAT);
    register_binary_kernel::<{ BinaryOp::ReluGrad as u32 }>(device_type, TF_DataType_TF_INT32);

    register_binary_kernel::<{ BinaryOp::RsqrtGrad as u32 }>(device_type, TF_DataType_TF_FLOAT);
    register_binary_kernel::<{ BinaryOp::RsqrtGrad as u32 }>(device_type, TF_DataType_TF_DOUBLE);

    if !ENV_SETTINGS.disable_int64 {
        register_binary_kernel::<{ BinaryOp::ReluGrad as u32 }>(device_type, TF_DataType_TF_INT64);
    }
}
