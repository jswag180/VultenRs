use backend::{memory::VultenCpyInfo, va::VaAddress, GOLBAL_DEVICE_VA};
use std::{ffi::c_void, mem::offset_of};
use tensorflow_pluggable_device_sys::{
    TF_AllocateOutput, TF_AllocateTemp, TF_AllocatorAttributes, TF_DataType, TF_DeleteStatus,
    TF_DeleteTensor, TF_Dim, TF_GetInput, TF_NewStatus, TF_NumDims, TF_OpKernelContext, TF_Tensor,
    TF_TensorData, TF_TensorElementCount, TF_TensorType, TSL_Code_TSL_OK, TSL_GetCode, TSL_Status,
};
use tracing::error;

use crate::{log_ops, stream::PluginStream};

#[derive(Debug)]
pub enum TensorData {
    Device(VaAddress),
    Host(*mut c_void),
}

#[derive(Debug)]
pub struct SafeTensor {
    pub tensor_ptr: *mut TF_Tensor,
    pub dims: Vec<i64>,
    pub d_type: TF_DataType,
    pub total_elements: i64,
    pub data: TensorData,
    pub is_empty: bool,
    pub is_scalar: bool,
}

impl SafeTensor {
    /// # Safety
    ///
    /// This function must be called with a valid TF_OpKernelContext ptr.
    pub unsafe fn from_input_device(
        input_num: i32,
        ctx: *mut TF_OpKernelContext,
        status: &SafeStatus,
    ) -> Self {
        let mut raw_tensor_ptr: *mut TF_Tensor = std::ptr::null_mut();
        TF_GetInput(ctx, input_num, &mut raw_tensor_ptr, status.status_ptr());

        if !status.is_ok() {
            error!("TF_GetInput return status {:?}", status.get_code());
            panic!();
        }

        // Dims
        let num_dims = unsafe { TF_NumDims(raw_tensor_ptr) } as usize;
        let mut dims: Vec<i64> = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            dims.push(TF_Dim(raw_tensor_ptr, i as i32));
        }

        // Type
        let d_type = TF_TensorType(raw_tensor_ptr);

        //Elements
        let total_elements = TF_TensorElementCount(raw_tensor_ptr);

        let data: TensorData = TensorData::Device(TF_TensorData(raw_tensor_ptr).into());

        let is_empty = total_elements <= 0;
        let is_scalar = total_elements == 1 && dims.is_empty();

        Self {
            tensor_ptr: raw_tensor_ptr,
            dims,
            d_type,
            total_elements,
            data,
            is_empty,
            is_scalar,
        }
    }

    /// # Safety
    ///
    /// This function must be called with a valid TF_OpKernelContext ptr.
    pub unsafe fn from_input_host(
        input_num: i32,
        ctx: *mut TF_OpKernelContext,
        status: &SafeStatus,
    ) -> Self {
        let mut raw_tensor_ptr: *mut TF_Tensor = std::ptr::null_mut();
        TF_GetInput(ctx, input_num, &mut raw_tensor_ptr, status.status_ptr());

        if !status.is_ok() {
            error!("TF_GetInput return status {:?}", status.get_code());
            panic!();
        }

        // Dims
        let num_dims = unsafe { TF_NumDims(raw_tensor_ptr) } as usize;
        let mut dims: Vec<i64> = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            dims.push(TF_Dim(raw_tensor_ptr, i as i32));
        }

        // Type
        let d_type = TF_TensorType(raw_tensor_ptr);

        //Elements
        let total_elements = TF_TensorElementCount(raw_tensor_ptr);

        if TF_TensorData(raw_tensor_ptr).is_null() {
            panic!("Tensor host ptr is null!");
        }
        let data: TensorData = TensorData::Host(TF_TensorData(raw_tensor_ptr));

        let is_empty = total_elements <= 0;
        let is_scalar = total_elements == 1 && dims.is_empty();

        Self {
            tensor_ptr: raw_tensor_ptr,
            dims,
            d_type,
            total_elements,
            data,
            is_empty,
            is_scalar,
        }
    }

    /// # Safety
    ///
    /// This function must be called with a valid TF_Tensor ptr.
    pub unsafe fn import_device(raw_tensor_ptr: *mut TF_Tensor) -> Self {
        // Dims
        let num_dims = unsafe { TF_NumDims(raw_tensor_ptr) } as usize;
        let mut dims: Vec<i64> = Vec::with_capacity(num_dims);
        for i in 0..num_dims {
            dims.push(TF_Dim(raw_tensor_ptr, i as i32));
        }

        // Type
        let d_type = TF_TensorType(raw_tensor_ptr);

        //Elements
        let total_elements = TF_TensorElementCount(raw_tensor_ptr);

        let data: TensorData = TensorData::Device(TF_TensorData(raw_tensor_ptr).into());

        let is_empty = total_elements <= 0;
        let is_scalar = total_elements == 1 && dims.is_empty();

        Self {
            tensor_ptr: raw_tensor_ptr,
            dims,
            d_type,
            total_elements,
            data,
            is_empty,
            is_scalar,
        }
    }

    /// # Safety
    ///
    /// This function must be called with a valid TF_OpKernelContext ptr.
    pub unsafe fn new_output_like(
        &self,
        output_num: i32,
        d_type: TF_DataType,
        ctx: *mut TF_OpKernelContext,
        status: &SafeStatus,
    ) -> Self {
        let raw_tensor_ptr = TF_AllocateOutput(
            ctx,
            output_num,
            d_type,
            self.dims.as_ptr(),
            self.dims.len() as i32,
            self.total_elements.try_into().unwrap(),
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!("TF_AllocateOutput return status {:?}", status.get_code());
            panic!();
        }

        let data: TensorData = TensorData::Device(TF_TensorData(raw_tensor_ptr).into());

        Self {
            tensor_ptr: raw_tensor_ptr,
            dims: self.dims.clone(),
            d_type,
            total_elements: self.total_elements,
            data,
            is_empty: self.is_empty,
            is_scalar: self.is_scalar,
        }
    }

    /// # Safety
    ///
    /// This function must be called with a valid TF_OpKernelContext ptr.
    pub unsafe fn new_output(
        output_num: i32,
        dims: Vec<i64>,
        d_type: TF_DataType,
        ctx: *mut TF_OpKernelContext,
        status: &SafeStatus,
    ) -> Self {
        let total_elements: i64 = dims.iter().product();
        let raw_tensor_ptr = TF_AllocateOutput(
            ctx,
            output_num,
            d_type,
            dims.as_ptr(),
            dims.len() as i32,
            total_elements.try_into().unwrap(),
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!("TF_AllocateOutput return status {:?}", status.get_code());
            panic!();
        }

        let data: TensorData = TensorData::Device(TF_TensorData(raw_tensor_ptr).into());

        let is_empty = total_elements <= 0;
        //this is a problem
        let is_scalar = total_elements == 1 && dims.is_empty();

        Self {
            tensor_ptr: raw_tensor_ptr,
            dims,
            d_type,
            total_elements,
            data,
            is_empty,
            is_scalar,
        }
    }

    /// # Safety
    ///
    /// This function must be called with a valid TF_OpKernelContext ptr.
    pub unsafe fn new_temp(
        dims: Vec<i64>,
        d_type: TF_DataType,
        ctx: *mut TF_OpKernelContext,
        status: &SafeStatus,
    ) -> Self {
        let total_elements: i64 = dims.iter().product();
        let mut alloc_attr: TF_AllocatorAttributes = TF_AllocatorAttributes {
            struct_size: offset_of!(TF_AllocatorAttributes, on_host) + std::mem::size_of::<u8>(),
            on_host: 0,
        };
        let raw_tensor_ptr = TF_AllocateTemp(
            ctx,
            d_type,
            dims.as_ptr(),
            dims.len() as i32,
            &mut alloc_attr,
            status.status_ptr(),
        );
        if !status.is_ok() {
            error!("TF_AllocateTemp return status {:?}", status.get_code());
            panic!();
        }

        let data: TensorData = TensorData::Device(TF_TensorData(raw_tensor_ptr).into());

        let is_empty = total_elements <= 0;
        let is_scalar = total_elements == 1 && dims.is_empty();

        Self {
            tensor_ptr: raw_tensor_ptr,
            dims,
            d_type,
            total_elements,
            data,
            is_empty,
            is_scalar,
        }
    }

    pub fn get_device_data(&self) -> Option<VaAddress> {
        if let TensorData::Device(addr) = self.data {
            Some(addr)
        } else {
            None
        }
    }

    pub fn get_host_data(&self) -> Option<*mut c_void> {
        if let TensorData::Host(addr) = self.data {
            Some(addr)
        } else {
            None
        }
    }
}

impl Drop for SafeTensor {
    fn drop(&mut self) {
        unsafe {
            if !self.tensor_ptr.is_null() {
                TF_DeleteTensor(self.tensor_ptr);
            }
        }
    }
}

pub struct SafeStatus(*mut TSL_Status);

impl SafeStatus {
    pub fn new() -> Self {
        unsafe { Self(TF_NewStatus()) }
    }

    pub fn status_ptr(&self) -> *mut TSL_Status {
        self.0
    }

    pub fn get_code(&self) -> u32 {
        unsafe { TSL_GetCode(self.0) }
    }

    #[inline]
    pub fn is_ok(&self) -> bool {
        self.get_code() == TSL_Code_TSL_OK
    }
}

impl Drop for SafeStatus {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                TF_DeleteStatus(self.0);
            }
        }
    }
}

impl Default for SafeStatus {
    fn default() -> Self {
        Self::new()
    }
}

/// # Safety
///
/// This function is ment for Tensorflow to call. So we have to trust it.
#[no_mangle]
pub unsafe extern "C" fn copy_func(
    ctx: *mut TF_OpKernelContext,
    src_tensor: *mut TF_Tensor,
    dst_tensor: *mut TF_Tensor,
) {
    let status = SafeStatus::new();

    let stream = unsafe { PluginStream::from_ctx(ctx, &status) };
    let inst = unsafe { &*stream.inst };

    let src = SafeTensor::import_device(src_tensor);
    let dst = SafeTensor::import_device(dst_tensor);

    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(src.get_device_data().unwrap())
    );
    debug_assert_eq!(
        inst.dev_num,
        VaAddress::get_device_num(dst.get_device_data().unwrap())
    );

    let (src_buffer, src_offset) = GOLBAL_DEVICE_VA
        .find_va(src.get_device_data().unwrap())
        .unwrap();
    let (dst_buffer, dst_offset) = GOLBAL_DEVICE_VA
        .find_va(dst.get_device_data().unwrap())
        .unwrap();

    log_ops!(
        "Running copy_func\n  Device: {:}\n  Stream: {:p}\n  Src: {:?} | {:?}\n  Dst: {:?} | {:?}",
        inst.dev_num,
        stream,
        src,
        src_offset,
        dst,
        dst_offset
    );

    let cpy_info = VultenCpyInfo::default()
        .src_offset(src_offset)
        .dst_offset(dst_offset)
        .size(src_buffer.obj.size);

    inst.blocking_cpy(src_buffer.obj.vk_buffer, dst_buffer.obj.vk_buffer, cpy_info);
}

#[derive(Debug)]
pub struct BroadcastShapeHelper {
    pub out_shape: Vec<i64>,
    pub a_padded: Vec<i64>,
    pub b_padded: Vec<i64>,
    pub needs_boardcast: bool,
    pub simple_boardcast: bool,
}

impl BroadcastShapeHelper {
    pub fn new(mut a_shape: Vec<i64>, mut b_shape: Vec<i64>) -> Result<Self, String> {
        let mut needs_boardcast = false;
        let mut simple_boardcast = true;

        if a_shape.is_empty() && b_shape.is_empty() {
            return Ok(Self {
                a_padded: a_shape,
                b_padded: b_shape,
                out_shape: Vec::new(),
                needs_boardcast: false,
                simple_boardcast: false,
            });
        }

        match a_shape.len().cmp(&b_shape.len()) {
            std::cmp::Ordering::Greater => {
                b_shape.reverse();
                b_shape.resize_with(a_shape.len(), || 1);
                b_shape.reverse();

                needs_boardcast = true;
            }
            std::cmp::Ordering::Less => {
                a_shape.reverse();
                a_shape.resize_with(b_shape.len(), || 1);
                a_shape.reverse();

                needs_boardcast = true;
            }
            _ => {
                for (a, b) in a_shape.iter().zip(b_shape.iter()) {
                    if a != b {
                        needs_boardcast = true;
                        break;
                    }
                }
            }
        }

        let mut out_shape = a_shape.clone();
        let mut a_leading_ones: usize = 0;
        let mut a_prev_not_one = false;
        let mut b_leading_ones: usize = 0;
        let mut b_prev_not_one = false;
        if needs_boardcast {
            for (indx, (a, b)) in a_shape.iter().zip(b_shape.iter()).enumerate() {
                if *a != *b {
                    if *a == 1 {
                        out_shape[indx] = *b;

                        if *b != 1 {
                            b_prev_not_one = true;
                        }
                        if !a_prev_not_one {
                            a_leading_ones += 1;
                        }
                    } else if *b == 1 {
                        //no need to set sense out_shape starts as a_shape
                        if !b_prev_not_one {
                            b_leading_ones += 1;
                        }
                        if *a != 1 {
                            a_prev_not_one = true;
                        }
                    } else {
                        return Err(format!(
                            "Shapes cannot be broadcast {:?} & {:?}",
                            a_shape, b_shape
                        ));
                    }
                } else if *a == 1 && *b == 1 {
                    if !b_prev_not_one {
                        b_leading_ones += 1;
                    }
                    if !a_prev_not_one {
                        a_leading_ones += 1;
                    }
                } else {
                    a_prev_not_one = true;
                    b_prev_not_one = true;
                }
            }
        }

        if a_leading_ones < b_leading_ones {
            for i in b_leading_ones..out_shape.len() {
                if a_shape[i] != b_shape[i] {
                    simple_boardcast = false;
                }
            }
        } else {
            for i in a_leading_ones..out_shape.len() {
                if a_shape[i] != b_shape[i] {
                    simple_boardcast = false;
                }
            }
        }

        if needs_boardcast && !simple_boardcast && a_shape.len() > 9 {
            return Err(format!("Max broadcast of 9 got {:}", a_shape.len()));
        }

        Ok(Self {
            out_shape,
            a_padded: a_shape,
            b_padded: b_shape,
            needs_boardcast,
            simple_boardcast: simple_boardcast && needs_boardcast,
        })
    }
}
