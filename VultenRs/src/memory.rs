use backend::{queue::VultenQueueFlags, va::VaAddress, GLOBAL_INSTANCES, GOLBAL_DEVICE_VA};
use std::os::raw::c_void;
use tensorflow_pluggable_device_sys::{SP_Device, SP_DeviceMemoryBase, SP_Stream, TF_Status};

use crate::{log_mem, profile, profile_add_stat};

//Async memcpys
#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_memcpy_dtoh(
    device: *const SP_Device,
    stream: SP_Stream,
    host_dst: *mut c_void,
    device_src: *const SP_DeviceMemoryBase,
    size: u64,
    _status: *mut TF_Status,
) {
    let inst = &mut *((*device).device_handle as *mut backend::VultenInstance);
    let mut prof = profile!("memcpy_dtoh".to_string(), inst.dev_num);
    profile_add_stat!(prof, "Size".to_string(), size.to_string());

    log_mem!(
        "opaque: {:?} mem Size: {:?}",
        (*device_src).opaque,
        (*device_src).size
    );

    let src: u64 = (*device_src).opaque as u64;
    let dst: u64 = host_dst as u64;

    (*(stream as *mut super::stream::PluginStream)).schedule_future(async move {
        let staging =
            inst.create_buffer(backend::memory::VultenBufferType::Host, size, false, true);
        let staging_ptr = staging.get_mapped_ptr().unwrap() as *mut u8;

        let src_buffer = GOLBAL_DEVICE_VA
            .find_va((src as *mut c_void).into())
            .unwrap();

        let cpy_info = backend::memory::VultenCpyInfo::default()
            .src_offset(src_buffer.1)
            .size(size);
        inst.blocking_cpy(src_buffer.0.obj.vk_buffer, staging.vk_buffer, cpy_info);

        unsafe { std::ptr::copy_nonoverlapping(staging_ptr, dst as *mut u8, size as usize) };
    });
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_memcpy_htod(
    device: *const SP_Device,
    stream: SP_Stream,
    device_dst: *mut SP_DeviceMemoryBase,
    host_src: *const c_void,
    size: u64,
    _status: *mut TF_Status,
) {
    let inst = &mut *((*device).device_handle as *mut backend::VultenInstance);
    let mut prof = profile!("memcpy_htod".to_string(), inst.dev_num);
    profile_add_stat!(prof, "Size".to_string(), size.to_string());

    log_mem!(
        "opaque: {:?} mem Size: {:?}",
        (*device_dst).opaque,
        (*device_dst).size
    );

    let src: u64 = host_src as u64;
    let dst: u64 = (*device_dst).opaque as u64;

    (*(stream as *mut super::stream::PluginStream)).schedule_future(async move {
        let staging = inst.create_buffer(
            backend::memory::VultenBufferType::Staging,
            size,
            true,
            false,
        );
        let staging_ptr = staging.get_mapped_ptr().unwrap() as *mut u8;
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, staging_ptr, size as usize) };

        let dst_buffer = GOLBAL_DEVICE_VA
            .find_va((dst as *mut c_void).into())
            .unwrap();

        let q = inst.get_queue(VultenQueueFlags::TRANSFER);
        let host_slice = std::slice::from_raw_parts(src as *const u8, size as usize);
        inst.upload_to_device_buff(host_slice, &dst_buffer.0.obj, dst_buffer.1, Some(&q))
            .unwrap();
    });
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_memcpy_dtod(
    device: *const SP_Device,
    stream: SP_Stream,
    device_dst: *mut SP_DeviceMemoryBase,
    device_src: *const SP_DeviceMemoryBase,
    size: u64,
    _status: *mut TF_Status,
) {
    let inst = &mut *((*device).device_handle as *mut backend::VultenInstance);
    let mut prof = profile!("memcpy_dtod".to_string(), inst.dev_num);
    profile_add_stat!(prof, "Size".to_string(), size.to_string());

    let src: u64 = (*device_src).opaque as u64;
    let dst: u64 = (*device_dst).opaque as u64;
    log_mem!(
        "src_opaque: {:p} dst_opaque: {:p}",
        (*device_src).opaque,
        (*device_dst).opaque
    );
    (*(stream as *mut super::stream::PluginStream)).schedule_future(async move {
        let (src_buffer, src_offset) = GOLBAL_DEVICE_VA
            .find_va((src as *mut c_void).into())
            .unwrap();
        let src_staging =
            inst.create_buffer(backend::memory::VultenBufferType::Host, size, true, true);
        let src_cpy_info = backend::memory::VultenCpyInfo::default()
            .src_offset(src_offset)
            .size(size);
        inst.blocking_cpy(
            src_buffer.obj.vk_buffer,
            src_staging.vk_buffer,
            src_cpy_info,
        );

        let dst_dev_num = VaAddress::get_device_num((dst as *mut c_void).into());
        let (dst_buffer, dst_offset) = GOLBAL_DEVICE_VA
            .find_va((dst as *mut c_void).into())
            .unwrap();

        let dst_inst = &(*GLOBAL_INSTANCES.read().unwrap()[dst_dev_num as usize]);
        let q = dst_inst.get_queue(VultenQueueFlags::TRANSFER);
        let host_slice = std::slice::from_raw_parts(
            src_staging.get_mapped_ptr().unwrap() as *const u8,
            size as usize,
        );
        dst_inst
            .upload_to_device_buff(host_slice, &dst_buffer.obj, dst_offset, Some(&q))
            .unwrap();
    });
}

//Sync memcpys
#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_sync_memcpy_dtoh(
    _device: *const SP_Device,
    host_dst: *mut c_void,
    device_src: *const SP_DeviceMemoryBase,
    size: u64,
    _status: *mut TF_Status,
) {
    log_mem!("sync dth s: {:?} d: {:?}", host_dst, (*device_src).opaque);
    todo!();
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_sync_memcpy_htod(
    _device: *const SP_Device,
    device_dst: *mut SP_DeviceMemoryBase,
    host_src: *const c_void,
    size: u64,
    _status: *mut TF_Status,
) {
    log_mem!("sync htd s: {:?} d: {:?}", (*device_dst).opaque, host_src);
    todo!();
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_sync_memcpy_dtod(
    _device: *const SP_Device,
    device_dst: *mut SP_DeviceMemoryBase,
    device_src: *const SP_DeviceMemoryBase,
    size: u64,
    _status: *mut TF_Status,
) {
    log_mem!(
        "sync dtd s: {:?} d: {:?}",
        (*device_dst).opaque,
        (*device_src).opaque
    );
    todo!();
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn pulgin_mem_zero(
    _device: *const SP_Device,
    stream: SP_Stream,
    location: *mut SP_DeviceMemoryBase,
    size: u64,
    _status: *mut TF_Status,
) {
    log_mem!(
        "mem zero buff: {:?} size: {:?} stream: {:?}",
        (*location).opaque,
        size,
        stream
    );
    todo!();
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn pulgin_memset(
    _device: *const SP_Device,
    stream: SP_Stream,
    location: *mut SP_DeviceMemoryBase,
    pattern: u8,
    size: u64,
    _status: *mut TF_Status,
) {
    log_mem!(
        "memset8 buff: {:?} patern: {:?} size: {:?} stream: {:?}",
        (*location).opaque,
        pattern,
        size,
        stream
    );
    todo!();
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn pulgin_memset32(
    _device: *const SP_Device,
    stream: SP_Stream,
    location: *mut SP_DeviceMemoryBase,
    pattern: u32,
    size: u64,
    _status: *mut TF_Status,
) {
    log_mem!(
        "memset32 buff: {:?} patern: {:?} size: {:?} stream: {:?}",
        (*location).opaque,
        pattern,
        size,
        stream
    );
    todo!();
}
