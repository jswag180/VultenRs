use std::ffi::c_char;

use backend::kernels::assign_add_sub_variable::{self, AssignOp};
use backend::va::VaAddress;
use backend::{ENV_SETTINGS, GOLBAL_DEVICE_VA};
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_AssignUpdateVariable, TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32,
    TF_DataType_TF_INT64, TF_DataType_TF_UINT32, TF_DataType_TF_UINT64,
    TF_KernelBuilder_TypeConstraint, TF_NewKernelBuilder, TF_OpKernelContext,
    TF_RegisterKernelBuilder, TF_Tensor,
};
use tracing::error;

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

use super::kernel_utills::copy_func;

#[no_mangle]
extern "C" fn update_func(
    ctx: *mut TF_OpKernelContext,
    tensor: *mut TF_Tensor,
    value: *mut TF_Tensor,
    op: i32,
) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };
    let _prof = profile!(
        format!("AddSub {:?}", <AssignOp>::try_from(op).unwrap()),
        inst.dev_num
    );

    let var = unsafe { SafeTensor::import_device(tensor) };
    let val = unsafe { SafeTensor::import_device(value) };

    log_ops!(
        "Running assign_add_sub_variable\n  Device: {:}\n  Stream: {:p}\n  Op: {:}\n  Tensor: {:?}\n  Value: {:?}",
        inst.dev_num,
        stream,
        op,
        var,
        val
    );

    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(var.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(val.get_device_data().unwrap())
    );

    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(var.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(val.get_device_data().unwrap())
        .is_ok());

    assign_add_sub_variable::AssignAddSubKernel::new(
        inst,
        var.d_type.into(),
        op.try_into().unwrap(),
    )
    .input(var.get_device_data().unwrap().into(), var.total_elements)
    .unwrap()
    .output(val.get_device_data().unwrap().into())
    .unwrap()
    .run()
    .unwrap();
}

#[no_mangle]
extern "C" fn compute_assign_add_variable(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    unsafe {
        TF_AssignUpdateVariable(
            ctx,
            0,
            1,
            0,
            0,
            Some(copy_func),
            Some(update_func),
            status.status_ptr(),
        );
    }
    if !status.is_ok() {
        error!(
            "TF_AssignUpdateVariable return status {:?}",
            status.get_code()
        );
        panic!();
    }
}

#[no_mangle]
extern "C" fn compute_assign_sub_variable(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    unsafe {
        TF_AssignUpdateVariable(
            ctx,
            0,
            1,
            1,
            0,
            Some(copy_func),
            Some(update_func),
            status.status_ptr(),
        );
    }
    if !status.is_ok() {
        error!(
            "TF_AssignUpdateVariable return status {:?}",
            status.get_code()
        );
        panic!();
    }
}

fn register_assign_add_sub_variable_kernel(
    device_type: *const c_char,
    d_type: TF_DataType,
    op: AssignOp,
) {
    let status = SafeStatus::new();

    let (builder, op_str) = match op {
        AssignOp::Add => unsafe {
            (
                TF_NewKernelBuilder(
                    c"AssignAddVariableOp".as_ptr(),
                    device_type,
                    None,
                    Some(compute_assign_add_variable),
                    None,
                ),
                c"AssignAddVariableOp".as_ptr(),
            )
        },
        AssignOp::Sub => unsafe {
            (
                TF_NewKernelBuilder(
                    c"AssignSubVariableOp".as_ptr(),
                    device_type,
                    None,
                    Some(compute_assign_sub_variable),
                    None,
                ),
                c"AssignSubVariableOp".as_ptr(),
            )
        },
    };

    unsafe {
        TF_KernelBuilder_TypeConstraint(builder, c"dtype".as_ptr(), d_type, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_KernelBuilder_TypeConstraint return status {:?}",
                status.get_code()
            );
            panic!();
        }

        TF_RegisterKernelBuilder(op_str, builder, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }
}

pub fn register_assign_add_sub_variable_op(device_type: *const c_char) {
    register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_FLOAT, AssignOp::Add);
    register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_FLOAT, AssignOp::Sub);

    register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_INT32, AssignOp::Add);
    register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_INT32, AssignOp::Sub);
    register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_UINT32, AssignOp::Add);
    register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_UINT32, AssignOp::Sub);

    if !ENV_SETTINGS.disable_int64 {
        register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_INT64, AssignOp::Add);
        register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_INT64, AssignOp::Sub);
        register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_UINT64, AssignOp::Add);
        register_assign_add_sub_variable_kernel(device_type, TF_DataType_TF_UINT64, AssignOp::Sub);
    }
}
