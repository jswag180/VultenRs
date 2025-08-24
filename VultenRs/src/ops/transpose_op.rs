use std::ffi::c_char;

use backend::dims::Dims;
use backend::kernels::transpose;
use backend::va::VaAddress;
use backend::{ENV_SETTINGS, GOLBAL_DEVICE_VA};
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32, TF_DataType_TF_INT64,
    TF_KernelBuilder_HostMemory, TF_KernelBuilder_TypeConstraint, TF_NewKernelBuilder,
    TF_OpKernelContext, TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

#[no_mangle]
extern "C" fn compute_transpose(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };
    let _prof = profile!("Transpose".to_string(), inst.dev_num);

    let input_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if input_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            input_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let paerm_tensor = unsafe { SafeTensor::from_input_host(1, ctx, &status) };
    let perm: Vec<i64> = if paerm_tensor.d_type == TF_DataType_TF_INT32 {
        let slice: &[i32] = unsafe {
            std::slice::from_raw_parts(
                paerm_tensor.get_host_data().unwrap() as *const i32,
                paerm_tensor.total_elements as usize,
            )
        };

        slice.to_vec().iter().map(|x| *x as i64).collect()
    } else if paerm_tensor.d_type == TF_DataType_TF_INT64 {
        let slice: &[i64] = unsafe {
            std::slice::from_raw_parts(
                paerm_tensor.get_host_data().unwrap() as *const _ as *const i64,
                paerm_tensor.total_elements as usize,
            )
        };

        slice.to_vec()
    } else {
        error!("Invalid perm type");
        panic!();
    };

    let mut output_dims = input_tensor.dims.clone();
    for i in 0..input_tensor.dims.len() {
        output_dims[i] = input_tensor.dims[perm[i] as usize];
    }
    let output_tensor =
        unsafe { SafeTensor::new_output(0, output_dims, input_tensor.d_type, ctx, &status) };

    log_ops!(
        "Running Transpose\n  Device: {:}\n  Stream: {:p}\n  Input: {:?}\n  Perm: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        input_tensor,
        perm,
        output_tensor
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

    transpose::TransposeKernel::new(inst, input_tensor.d_type.into())
        .input(
            input_tensor.get_device_data().unwrap().into(),
            &input_tensor.dims,
        )
        .unwrap()
        .transpose(Dims::Slice(&perm))
        .unwrap()
        .output(
            output_tensor.get_device_data().unwrap().into(),
            Dims::Slice(&output_tensor.dims),
        )
        .unwrap()
        .run()
        .unwrap();
}

fn register_transpose_kernel(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let builder = unsafe {
        TF_NewKernelBuilder(
            c"Transpose".as_ptr(),
            device_type,
            None,
            Some(compute_transpose),
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

        TF_KernelBuilder_HostMemory(builder, c"perm".as_ptr());

        TF_RegisterKernelBuilder(c"Transpose".as_ptr(), builder, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }
}

pub fn register_transpose_op(device_type: *const c_char) {
    register_transpose_kernel(device_type, TF_DataType_TF_FLOAT);
    register_transpose_kernel(device_type, TF_DataType_TF_INT32);
    if !ENV_SETTINGS.disable_int64 {
        register_transpose_kernel(device_type, TF_DataType_TF_INT64);
    }
}
