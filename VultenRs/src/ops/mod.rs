pub mod kernel_utills;

pub mod addn_op;
pub mod assign_add_sub_variable_op;
pub mod bias_add_grad_op;
pub mod bias_add_op;
pub mod binary_ops;
pub mod conv2d_backprop_filter_op;
pub mod conv2d_backprop_input_op;
pub mod conv2d_op;
pub mod matmul_op;
pub mod reduce_ops;
pub mod sscel;
pub mod transpose_op;
pub mod unary_ops;

use crate::DEVICE_TYPE;

#[no_mangle]
pub extern "C" fn TF_InitKernel() {
    assign_add_sub_variable_op::register_assign_add_sub_variable_op(DEVICE_TYPE);
    binary_ops::register_binary_ops(DEVICE_TYPE);
    unary_ops::register_unary_ops(DEVICE_TYPE);
    matmul_op::register_matmul_op(DEVICE_TYPE);
    bias_add_op::register_bias_add_op(DEVICE_TYPE);
    addn_op::register_addn_op(DEVICE_TYPE);
    reduce_ops::register_reduce_ops(DEVICE_TYPE);
    bias_add_grad_op::register_bias_add_grad_op(DEVICE_TYPE);
    sscel::register_sscel_op(DEVICE_TYPE);
    conv2d_op::register_conv2d_op(DEVICE_TYPE);
    conv2d_backprop_filter_op::register_conv2d_backprop_filter_op(DEVICE_TYPE);
    conv2d_backprop_input_op::register_conv2d_backprop_input_op(DEVICE_TYPE);
    transpose_op::register_transpose_op(DEVICE_TYPE);
}
