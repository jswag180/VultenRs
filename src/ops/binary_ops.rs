use backend::kernels::binary::{self, BinaryOp};
use backend::kernels::KernelInput;
use backend::va::VaAddress;
use backend::GOLBAL_DEVICE_VA;
use libc::c_void;
use tensorflow_pluggable_device_sys::{
    TF_DataType, TF_DataType_TF_FLOAT, TF_DataType_TF_INT32, TF_DataType_TF_INT64,
    TF_DataType_TF_UINT32, TF_DataType_TF_UINT64, TF_KernelBuilder_TypeConstraint,
    TF_NewKernelBuilder, TF_OpKernelContext, TF_RegisterKernelBuilder,
};
use tracing::error;

use crate::log_ops;
use crate::ops::kernel_utills::{BroadcastShapeHelper, SafeStatus, SafeTensor};
use crate::stream::PluginStream;

//#[no_mangle]
extern "C" fn compute_binary<const T: u32>(_info: *mut c_void, ctx: *mut TF_OpKernelContext) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };

    let x_tensor = unsafe { SafeTensor::from_input(0, ctx, &status) };
    if x_tensor.total_elements > u32::MAX as i64 {
        error!(
            "Input tensor is to big {:} > {:}",
            x_tensor.total_elements,
            u32::MAX
        );
        return;
    }

    let y_tensor = unsafe { SafeTensor::from_input(1, ctx, &status) };
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

    let output_tensor =
        unsafe { SafeTensor::new_output(0, shape_helper.out_shape, x_tensor.d_type, ctx, &status) };

    if x_tensor.is_empty || y_tensor.is_empty {
        return;
    }

    debug_assert_eq!(inst.dev_num, VaAddress::get_device_num(x_tensor.data));
    debug_assert_eq!(inst.dev_num, VaAddress::get_device_num(y_tensor.data));
    debug_assert_eq!(inst.dev_num, VaAddress::get_device_num(output_tensor.data));

    unsafe {
        debug_assert!(GOLBAL_DEVICE_VA.find_va(x_tensor.data).is_ok());
        debug_assert!(GOLBAL_DEVICE_VA.find_va(y_tensor.data).is_ok());
        debug_assert!(GOLBAL_DEVICE_VA.find_va(output_tensor.data).is_ok());
    }

    if !shape_helper.needs_boardcast {
        binary::binary_no_board::run(
            inst,
            x_tensor.d_type.into(),
            <u32 as TryInto<BinaryOp>>::try_into(T).unwrap(),
            x_tensor.data,
            y_tensor.data,
            output_tensor.data,
            output_tensor.total_elements,
        )
        .unwrap();
    } else if shape_helper.simple_boardcast {
        binary::binary_simple::run(
            inst,
            x_tensor.d_type.into(),
            <u32 as TryInto<BinaryOp>>::try_into(T).unwrap(),
            x_tensor.data,
            x_tensor.total_elements,
            y_tensor.data,
            y_tensor.total_elements,
            output_tensor.data,
        )
        .unwrap();
    } else {
        let x = KernelInput {
            addr: x_tensor.data,
            dims: &shape_helper.a_padded,
        };
        let y = KernelInput {
            addr: y_tensor.data,
            dims: &shape_helper.b_padded,
        };
        let output = KernelInput {
            addr: output_tensor.data,
            dims: &output_tensor.dims,
        };
        binary::binary_broad::run(
            inst,
            x_tensor.d_type.into(),
            <u32 as TryInto<BinaryOp>>::try_into(T).unwrap(),
            x,
            y,
            output,
        )
        .unwrap();
    }
}

fn register_binary_kernel<const T: u32>(device_type: *const i8, d_type: TF_DataType) {
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
fn register_type(device_type: *const i8, d_type: TF_DataType) {
    register_binary_kernel::<{ BinaryOp::Mul.into_u32() }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Add.into_u32() }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Sub.into_u32() }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Div.into_u32() }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Div.into_u32() }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::DivNoNan.into_u32() }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::DivReal.into_u32() }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Max.into_u32() }>(device_type, d_type);
    register_binary_kernel::<{ BinaryOp::Min.into_u32() }>(device_type, d_type);
}

pub fn register_binary_ops(device_type: *const i8) {
    register_type(device_type, TF_DataType_TF_FLOAT);
    register_type(device_type, TF_DataType_TF_INT32);
    register_type(device_type, TF_DataType_TF_UINT32);
    register_type(device_type, TF_DataType_TF_INT64);
    register_type(device_type, TF_DataType_TF_UINT64);
}
