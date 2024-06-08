use std::ffi::c_char;

use backend::kernels::reduce::{self, process_dims, ReduceOp};
use backend::kernels::KernelInput;
use backend::memory::VultenCpyInfo;
use backend::va::VaAddress;
use backend::GOLBAL_DEVICE_VA;
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32, TF_DataType_TF_INT64,
    TF_DataType_TF_UINT32, TF_DataType_TF_UINT64, TF_KernelBuilder_HostMemory,
    TF_KernelBuilder_TypeConstraint, TF_NewKernelBuilder, TF_OpKernelConstruction,
    TF_OpKernelConstruction_GetAttrBool, TF_OpKernelContext, TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::log_ops;
use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;

#[derive(Debug, Default)]
#[repr(C)]
struct ReduceInfo {
    keep_dims: bool,
}

#[no_mangle]
extern "C" fn create_reduce(ctx: *mut TF_OpKernelConstruction) -> *mut c_void {
    let mut info = Box::<ReduceInfo>::default();

    let status = SafeStatus::new();
    unsafe {
        TF_OpKernelConstruction_GetAttrBool(
            ctx,
            c"keep_dims".as_ptr(),
            &mut info.keep_dims as *mut bool as *mut u8,
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!(
                "TF_OpKernelConstruction_GetAttrBool for keep_dims return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }

    Box::leak(info) as *mut ReduceInfo as *mut c_void
}

//#[no_mangle]
extern "C" fn compute_reduce<const T: u32>(info_ptr: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let info: &ReduceInfo = unsafe { &*(info_ptr as *const ReduceInfo) };

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };

    let input_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if input_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            input_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let reduce_dims_tensor = unsafe { SafeTensor::from_input_host(1, ctx, &status) };
    if reduce_dims_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Reduce dims tensor is to big {:} > {:}",
            reduce_dims_tensor.total_elements,
            u32::MAX
        );
        return;
    }
    let mut reduce_dims: Vec<u32> = if reduce_dims_tensor.d_type == TF_DataType_TF_INT32 {
        let reduce_dims_slice: &[i32] = unsafe {
            std::slice::from_raw_parts(
                reduce_dims_tensor.get_host_data().unwrap() as *const i32,
                reduce_dims_tensor.total_elements as usize,
            )
        };
        process_dims(&input_tensor.dims, reduce_dims_slice).unwrap()
    } else if reduce_dims_tensor.d_type == TF_DataType_TF_INT64 {
        let reduce_dims_slice: &[i64] = unsafe {
            std::slice::from_raw_parts(
                reduce_dims_tensor.get_host_data().unwrap() as *const _ as *const i64,
                reduce_dims_tensor.total_elements as usize,
            )
        };
        process_dims(&input_tensor.dims, reduce_dims_slice).unwrap()
    } else {
        Vec::new()
    };

    let mut output_dims: Vec<i64> = input_tensor.dims.clone();
    if !reduce_dims.is_empty() {
        if info.keep_dims {
            for reduce_dim in &reduce_dims {
                output_dims[*reduce_dim as usize] = 1;
            }
        } else {
            for i in (0..reduce_dims.len()).rev() {
                output_dims.remove(reduce_dims[i] as usize);
            }
        }
    }
    let output_tensor =
        unsafe { SafeTensor::new_output(0, output_dims, input_tensor.d_type, ctx, &status) };

    log_ops!(
        "Running Reduce\n  Device: {:}\n  Stream: {:p}\n  Type: {:?}\n  Op: {:?}\n  Dims: {:?}\n  Keep dims: {:?}\n  Input: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        input_tensor.d_type,
        <u32 as TryInto<ReduceOp>>::try_into(T).unwrap(),
        reduce_dims,
        info.keep_dims,
        input_tensor,
        output_tensor,
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

    unsafe {
        debug_assert!(GOLBAL_DEVICE_VA
            .find_va(input_tensor.get_device_data().unwrap())
            .is_ok());
        debug_assert!(GOLBAL_DEVICE_VA
            .find_va(output_tensor.get_device_data().unwrap())
            .is_ok());
    }

    if input_tensor.is_scalar || reduce_dims_tensor.is_empty {
        let input_buff = unsafe {
            GOLBAL_DEVICE_VA
                .find_va(input_tensor.get_device_data().unwrap())
                .unwrap()
        };
        let output_buff = unsafe {
            GOLBAL_DEVICE_VA
                .find_va(output_tensor.get_device_data().unwrap())
                .unwrap()
        };
        let cpy_info = VultenCpyInfo {
            src_offset: output_buff.1,
            dst_offset: input_buff.1,
            size: input_buff.0.size,
        };

        inst.blocking_cpy(
            input_buff.0.obj.vk_buffer,
            output_buff.0.obj.vk_buffer,
            cpy_info,
        );
        return;
    }

    let input = KernelInput {
        addr: input_tensor.get_device_data().unwrap(),
        dims: &input_tensor.dims,
    };
    let output = KernelInput {
        addr: output_tensor.get_device_data().unwrap(),
        dims: &output_tensor.dims,
    };

    reduce_dims.reverse();

    reduce::reduce::run(
        inst,
        input_tensor.d_type.into(),
        <u32 as TryInto<ReduceOp>>::try_into(T).unwrap(),
        reduce_dims,
        input,
        output,
    )
    .unwrap();
}

#[no_mangle]
extern "C" fn destroy_reduce(info: *mut c_void) {
    let info_box: Box<ReduceInfo> = unsafe { Box::from_raw(info as *mut ReduceInfo) };
    drop(info_box);
}

fn register_reduce_kernel<const T: u32>(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let op_str = match T.try_into().unwrap() {
        ReduceOp::Sum => c"Sum",
        ReduceOp::Max => c"Max",
        ReduceOp::Min => c"Min",
    };

    let builder = unsafe {
        TF_NewKernelBuilder(
            op_str.as_ptr(),
            device_type,
            Some(create_reduce),
            Some(compute_reduce::<T>),
            Some(destroy_reduce),
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

        TF_KernelBuilder_HostMemory(builder, c"reduction_indices".as_ptr());

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
    register_reduce_kernel::<{ ReduceOp::Sum.into_u32() }>(device_type, d_type);
    register_reduce_kernel::<{ ReduceOp::Max.into_u32() }>(device_type, d_type);
    register_reduce_kernel::<{ ReduceOp::Min.into_u32() }>(device_type, d_type);
}

pub fn register_reduce_ops(device_type: *const c_char) {
    register_type(device_type, TF_DataType_TF_FLOAT);
    register_type(device_type, TF_DataType_TF_INT32);
    register_type(device_type, TF_DataType_TF_UINT32);
    register_type(device_type, TF_DataType_TF_INT64);
    register_type(device_type, TF_DataType_TF_UINT64);
}
