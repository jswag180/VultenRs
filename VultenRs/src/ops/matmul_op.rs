use std::ffi::c_char;

use backend::kernels::{matmul, KernelInput};
use backend::va::VaAddress;
use backend::GOLBAL_DEVICE_VA;
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32, TF_DataType_TF_INT64,
    TF_DataType_TF_UINT32, TF_DataType_TF_UINT64, TF_KernelBuilder_TypeConstraint,
    TF_NewKernelBuilder, TF_OpKernelConstruction, TF_OpKernelConstruction_GetAttrBool,
    TF_OpKernelContext, TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::ops::kernel_utills::{SafeStatus, SafeTensor};
use crate::stream::PluginStream;
use crate::{log_ops, profile};

pub const INLINE_CUTOFF: i64 = 4096;

#[derive(Debug, Default)]
#[repr(C)]
struct MatmulInfo {
    trans_a: bool,
    trans_b: bool,
}

#[no_mangle]
extern "C" fn create_matmul(ctx: *mut TF_OpKernelConstruction) -> *mut c_void {
    let mut info = Box::<MatmulInfo>::default();

    let status = SafeStatus::new();
    unsafe {
        TF_OpKernelConstruction_GetAttrBool(
            ctx,
            c"transpose_a".as_ptr(),
            &mut info.trans_a as *mut bool as *mut u8,
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!(
                "TF_OpKernelConstruction_GetAttrBool for transpose_a return status {:?}",
                status.get_code()
            );
            panic!();
        }

        TF_OpKernelConstruction_GetAttrBool(
            ctx,
            c"transpose_b".as_ptr(),
            &mut info.trans_b as *mut bool as *mut u8,
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!(
                "TF_OpKernelConstruction_GetAttrBool for transpose_b return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }

    Box::leak(info) as *mut MatmulInfo as *mut c_void
}

#[no_mangle]
extern "C" fn compute_matmul(info_ptr: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let info: &MatmulInfo = unsafe { &*(info_ptr as *const MatmulInfo) };

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };
    let _prof = profile!("MatMul".to_string(), inst.dev_num);

    let a_tensor = unsafe { SafeTensor::from_input_device(0, ctx, &status) };
    if a_tensor.total_elements > u32::MAX as i64 {
        error!(
            "A tensor is to big {:} > {:}",
            a_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let b_tensor = unsafe { SafeTensor::from_input_device(1, ctx, &status) };
    if b_tensor.total_elements > u32::MAX as i64 {
        error!(
            "B tensor is to big {:} > {:}",
            b_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let mat_a_post: (i64, i64) = if info.trans_a {
        (a_tensor.dims[1], a_tensor.dims[0])
    } else {
        (a_tensor.dims[0], a_tensor.dims[1])
    };
    let mat_b_post: (i64, i64) = if info.trans_b {
        (b_tensor.dims[1], b_tensor.dims[0])
    } else {
        (b_tensor.dims[0], b_tensor.dims[1])
    };

    if mat_a_post.1 != mat_b_post.0 {
        error!(
            "Matrix {:?} is incompatable with {:?}",
            mat_a_post, mat_b_post
        );
        return;
    }

    let out_dims: Vec<i64> = vec![mat_a_post.0, mat_b_post.1];
    let output_tensor =
        unsafe { SafeTensor::new_output(0, out_dims, a_tensor.d_type, ctx, &status) };

    let not_inline =
        (output_tensor.total_elements > INLINE_CUTOFF) && (info.trans_a || info.trans_b);

    log_ops!(
        "Running MatMul\n  Device: {:}\n  Stream: {:p}\n  Info: {:?}\n  Inline: {:?}\n  A: {:?}\n  B: {:?}\n  Output: {:?}",
        inst.dev_num,
        stream,
        info,
        !not_inline,
        a_tensor,
        b_tensor,
        output_tensor,
    );

    if a_tensor.is_empty || b_tensor.is_empty {
        return;
    }

    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(a_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(b_tensor.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(output_tensor.get_device_data().unwrap())
    );

    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(a_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(b_tensor.get_device_data().unwrap())
        .is_ok());
    debug_assert!(GOLBAL_DEVICE_VA
        .find_va(output_tensor.get_device_data().unwrap())
        .is_ok());

    let a = KernelInput {
        buff: a_tensor.get_device_data().unwrap().into(),
        dims: &a_tensor.dims,
    };
    let b = KernelInput {
        buff: b_tensor.get_device_data().unwrap().into(),
        dims: &b_tensor.dims,
    };
    let output = KernelInput {
        buff: output_tensor.get_device_data().unwrap().into(),
        dims: &output_tensor.dims,
    };

    if not_inline {
        matmul::matmul::run(
            inst,
            a_tensor.d_type.into(),
            &a,
            info.trans_a,
            &b,
            info.trans_b,
            &output,
        )
        .unwrap();
    } else {
        matmul::matmul_inline_transpose::run(
            inst,
            a_tensor.d_type.into(),
            &a,
            info.trans_a,
            &b,
            info.trans_b,
            &output,
        )
        .unwrap();
    }
}

#[no_mangle]
extern "C" fn destroy_matmul(info: *mut c_void) {
    let info_box: Box<MatmulInfo> = unsafe { Box::from_raw(info as *mut MatmulInfo) };
    drop(info_box);
}

fn register_matmul_kernel(device_type: *const c_char, d_type: TF_DataType) {
    let status = SafeStatus::new();

    let builder = unsafe {
        TF_NewKernelBuilder(
            c"MatMul".as_ptr(),
            device_type,
            Some(create_matmul),
            Some(compute_matmul),
            Some(destroy_matmul),
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

        TF_RegisterKernelBuilder(c"MatMul".as_ptr(), builder, status.status_ptr());
        if !status.is_ok() {
            error!(
                "TF_RegisterKernelBuilder return status {:?}",
                status.get_code()
            );
            panic!();
        }
    }
}

pub fn register_matmul_op(device_type: *const c_char) {
    register_matmul_kernel(device_type, TF_DataType_TF_FLOAT);
    register_matmul_kernel(device_type, TF_DataType_TF_INT32);
    register_matmul_kernel(device_type, TF_DataType_TF_UINT32);
    register_matmul_kernel(device_type, TF_DataType_TF_INT64);
    register_matmul_kernel(device_type, TF_DataType_TF_UINT64);
}
