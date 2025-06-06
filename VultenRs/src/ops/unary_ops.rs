use std::ffi::c_char;

use backend::kernels::unary::{self, UnaryOp};
use backend::va::VaAddress;
use backend::{ENV_SETTINGS, GOLBAL_DEVICE_VA};
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32, TF_DataType_TF_INT64,
    TF_DataType_TF_UINT32, TF_DataType_TF_UINT64, TF_KernelBuilder_TypeConstraint,
    TF_NewKernelBuilder, TF_OpKernelContext, TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

//#[no_mangle]
extern "C" fn compute_unary<const T: u32>(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };
    let _prof = profile!(
        format!("{:?}", <u32 as TryInto<UnaryOp>>::try_into(T).unwrap()),
        inst.dev_num
    );

    let input_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if input_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            input_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    if input_tensor.is_empty {
        return;
    }

    let output_tensor =
        unsafe { input_tensor.new_output_like(0, input_tensor.d_type, ctx, &status) };

    log_ops!(
        "Running Unary\n  Device: {:}\n  Stream: {:p}\n  Type: {:?}\n  Op: {:?}\n  Input: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        input_tensor.d_type,
        <u32 as TryInto<UnaryOp>>::try_into(T).unwrap(),
        input_tensor,
        output_tensor,
    );

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

    unary::UnaryKernel::new(
        inst,
        input_tensor.d_type.into(),
        <u32 as TryInto<UnaryOp>>::try_into(T).unwrap(),
    )
    .input(
        input_tensor.get_device_data().unwrap().into(),
        input_tensor.total_elements,
    )
    .unwrap()
    .output(output_tensor.get_device_data().unwrap().into())
    .unwrap()
    .run()
    .unwrap();
}

fn register_unary_kernel<const T: u32>(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let op_str = match T.try_into().unwrap() {
        UnaryOp::Sqrt => c"Sqrt",
        UnaryOp::Exp => c"Exp",
        UnaryOp::Log => c"Log",
        UnaryOp::Square => c"Square",
        UnaryOp::Neg => c"Neg",
        UnaryOp::Reciprocal => c"Reciprocal",
        UnaryOp::Log1p => c"Log1p",
        UnaryOp::Tanh => c"Tanh",
        UnaryOp::Relu => c"Relu",
        UnaryOp::Rsqrt => c"Rsqrt",
    };

    let builder = unsafe {
        TF_NewKernelBuilder(
            op_str.as_ptr(),
            device_type,
            None,
            Some(compute_unary::<T>),
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
}

#[inline(always)]
fn register_type(device_type: *const c_char, d_type: TF_DataType) {
    register_unary_kernel::<{ UnaryOp::Sqrt as u32 }>(device_type, d_type);
    register_unary_kernel::<{ UnaryOp::Exp as u32 }>(device_type, d_type);
    register_unary_kernel::<{ UnaryOp::Log as u32 }>(device_type, d_type);
    register_unary_kernel::<{ UnaryOp::Square as u32 }>(device_type, d_type);
    register_unary_kernel::<{ UnaryOp::Neg as u32 }>(device_type, d_type);
    register_unary_kernel::<{ UnaryOp::Reciprocal as u32 }>(device_type, d_type);
    register_unary_kernel::<{ UnaryOp::Log1p as u32 }>(device_type, d_type);
    register_unary_kernel::<{ UnaryOp::Rsqrt as u32 }>(device_type, d_type);
}

pub fn register_unary_ops(device_type: *const c_char) {
    register_type(device_type, TF_DataType_TF_FLOAT);
    register_type(device_type, TF_DataType_TF_INT32);
    register_type(device_type, TF_DataType_TF_UINT32);
    if !ENV_SETTINGS.disable_int64 {
        register_type(device_type, TF_DataType_TF_INT64);
        register_type(device_type, TF_DataType_TF_UINT64);
    }

    register_unary_kernel::<{ UnaryOp::Tanh as u32 }>(device_type, TF_DataType_TF_FLOAT);

    register_unary_kernel::<{ UnaryOp::Relu as u32 }>(device_type, TF_DataType_TF_FLOAT);
    register_unary_kernel::<{ UnaryOp::Relu as u32 }>(device_type, TF_DataType_TF_INT32);

    if !ENV_SETTINGS.disable_int64 {
        register_unary_kernel::<{ UnaryOp::Relu as u32 }>(device_type, TF_DataType_TF_INT64);
    }
}
