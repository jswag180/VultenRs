use std::{mem::offset_of, sync::Arc};

use backend::{memory::VultenBufferType, GOLBAL_DEVICE_VA};
use libc::c_void;
use tensorflow_pluggable_device_sys::{SP_AllocatorStats, SP_Device, SP_DeviceMemoryBase, TF_Bool};
use tracing::debug;

use crate::{log_mem, profile, profile_add_stat};

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_allocate(
    device: *const SP_Device,
    size: u64,
    _memory_space: i64,
    mem: *mut SP_DeviceMemoryBase,
) {
    let inst = &*((*device).device_handle as *mut backend::VultenInstance);
    let mut prof = profile!("Allocate".to_string(), inst.dev_num);

    let dumb_struct: core::mem::MaybeUninit<SP_DeviceMemoryBase> = core::mem::MaybeUninit::uninit();
    (*mem).struct_size = offset_of!(SP_DeviceMemoryBase, payload)
        + std::mem::size_of_val(&dumb_struct.assume_init().payload); // This is a re-impl of the SP_DEVICE_MEMORY_BASE_STRUCT_SIZE macro

    let addr = GOLBAL_DEVICE_VA
        .alloc(
            (*device).ordinal as u64,
            Arc::new(inst.create_buffer(VultenBufferType::Device, size, true, true)),
            size,
        )
        .unwrap();
    (*mem).opaque = addr.raw_ptr();

    (*mem).size = size;
    profile_add_stat!(prof, "Size".to_string(), size.to_string());
    log_mem!("opaque: {:?} mem size: {:?}", (*mem).opaque, (*mem).size);
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_deallocate(
    device: *const SP_Device,
    mem: *mut SP_DeviceMemoryBase,
) {
    let inst = &*((*device).device_handle as *mut backend::VultenInstance);
    let mut prof = profile!("Deallocate".to_string(), inst.dev_num);
    profile_add_stat!(prof, "Size".to_string(), (*mem).size.to_string());

    log_mem!("opaque: {:?} mem size: {:?}", (*mem).opaque, (*mem).size);

    (*mem).size = 0;
    GOLBAL_DEVICE_VA.free((*mem).opaque.into()).unwrap();
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_get_allocator_stats(
    _device: *const SP_Device,
    stats: *mut SP_AllocatorStats,
) -> TF_Bool {
    let dumb_struct: core::mem::MaybeUninit<SP_AllocatorStats> = core::mem::MaybeUninit::uninit();
    (*stats).struct_size = offset_of!(SP_AllocatorStats, largest_free_block_bytes)
        + std::mem::size_of_val(&dumb_struct.assume_init().largest_free_block_bytes); // This is a re-impl of the SP_ALLOCATORSTATS_STRUCT_SIZE macro
    (*stats).bytes_in_use = 1;

    log_mem!("");
    true as TF_Bool
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_host_memory_allocate(
    _device: *const SP_Device,
    size: u64,
) -> *mut c_void {
    log_mem!("");
    libc::memalign(64, size as usize)
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_host_memory_deallocate(
    _device: *const SP_Device,
    mem_ptr: *mut c_void,
) {
    log_mem!("");
    libc::free(mem_ptr)
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_device_memory_usage(
    device: *const SP_Device,
    free: *mut i64,
    total: *mut i64,
) -> TF_Bool {
    let inst = (*device).device_handle as *mut backend::VultenInstance;

    let stats = (*inst).get_mem_stats();
    *free = (stats.0 as f64 * 0.9) as i64;
    *total = (stats.1 as f64 * 0.9) as i64;

    log_mem!("Total: {:?} Free: {:?}", *total, *free);

    true as TF_Bool
}
