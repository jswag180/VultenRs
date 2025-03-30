use std::ffi::c_char;

use backend::kernels::assign_add_sub_variable::{self, AssignOp};
use backend::va::VaAddress;
use backend::{ENV_SETTINGS, GOLBAL_DEVICE_VA};
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32, TF_DataType_TF_INT64,
    TF_DataType_TF_UINT32, TF_DataType_TF_UINT64, TF_KernelBuilder_TypeConstraint,
    TF_NewKernelBuilder, TF_NumInputs, TF_OpKernelContext, TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

#[no_mangle]
extern "C" fn compute_addn(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };
    let _prof = profile!("AddN".to_string(), inst.dev_num);

    let input_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if input_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            input_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(input_tensor.get_device_data().unwrap())
    );
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(input_tensor.get_device_data().unwrap())
        .is_ok());

    let num_tensors = unsafe { TF_NumInputs(ctx) };
    let mut input_tensors: Vec<SafeTensor> = Vec::with_capacity(num_tensors as usize);
    input_tensors.push(input_tensor);
    for i in 1..num_tensors {
        let new_tensor = unsafe { SafeTensor::from_input_device(i, ctx, &status) };
        debug_assert_eq!(
            inst.dev_num,
            VaAddress::get_device_num(new_tensor.get_device_data().unwrap())
        );
        debug_assert!(GOLBAL_DEVICE_VA
            .find_va(new_tensor.get_device_data().unwrap())
            .is_ok());

        if input_tensors[0].dims != new_tensor.dims {
            error!("All tensors provided to AddN must be the same shape");
            return;
        }

        input_tensors.push(new_tensor);
    }

    let output_tensor =
        unsafe { input_tensors[0].new_output_like(0, input_tensors[0].d_type, ctx, &status) };

    log_ops!(
        "Running AddN\n  Device: {:}\n  Stream: {:p}\n  Inputs: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        input_tensors,
        output_tensor
    );

    if input_tensors[0].is_empty {
        return;
    }

    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(output_tensor.get_device_data().unwrap())
    );

    let out_buff = GOLBAL_DEVICE_VA
        .find_va(output_tensor.get_device_data().unwrap())
        .unwrap();
    inst.fill_buffer(&out_buff.0.obj, out_buff.0.size, out_buff.1, 0)
        .unwrap();

    for tensor in input_tensors {
        assign_add_sub_variable::AssignAddSubKernel::new(
            inst,
            output_tensor.d_type.into(),
            AssignOp::Add,
        )
        .input(
            output_tensor.get_device_data().unwrap().into(),
            output_tensor.total_elements,
        )
        .unwrap()
        .output(tensor.get_device_data().unwrap().into())
        .unwrap()
        .run()
        .unwrap();
    }
}

fn register_addn_kernel(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let builder = unsafe {
        TF_NewKernelBuilder(
            c"AddN".as_ptr(),
            device_type,
            None,
            Some(compute_addn),
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

        TF_RegisterKernelBuilder(c"AddN".as_ptr(), builder, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }
}

pub fn register_addn_op(device_type: *const c_char) {
    register_addn_kernel(device_type, TF_DataType_TF_FLOAT);
    register_addn_kernel(device_type, TF_DataType_TF_INT32);
    register_addn_kernel(device_type, TF_DataType_TF_UINT32);
    if !ENV_SETTINGS.disable_int64 {
        register_addn_kernel(device_type, TF_DataType_TF_INT64);
        register_addn_kernel(device_type, TF_DataType_TF_UINT64);
    }
}
