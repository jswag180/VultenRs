use backend::GLOBAL_INSTANCES;
use std::{ffi::c_void, mem::offset_of};
use tensorflow_pluggable_device_sys::{
    SE_CreateDeviceFnsParams, SE_CreateDeviceParams, SE_CreateStreamExecutorParams,
    SE_PlatformRegistrationParams, SE_StatusCallbackFn, SP_Device, SP_DeviceFns, SP_Platform,
    SP_PlatformFns, SP_Stream, SP_StreamExecutor, TF_Bool, TF_SetStatus, TF_Status,
    TSL_Code_TSL_OK,
};
use tracing::info;

pub mod log;

pub mod allocaters;
pub mod event;
pub mod memory;
pub mod stream;

pub mod ops;

pub const DEVICE_NAME: *const i8 = c"VULTEN".as_ptr();
pub const DEVICE_TYPE: *const i8 = c"VULK".as_ptr();

#[tracing::instrument]
unsafe extern "C" fn plug_get_device_count(
    _platform: *const SP_Platform,
    count: *mut i32,
    _status: *mut TF_Status,
) {
    *count = backend::VultenInstance::get_num_devices();
    GLOBAL_INSTANCES
        .write()
        .unwrap()
        .reserve_exact(*count as usize);
    log_init!("Device Count {}", *count);
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_create_device(
    _platform: *const SP_Platform,
    params: *mut SE_CreateDeviceParams,
    _status: *mut TF_Status,
) {
    let device = &mut *(*params).device;

    let dumb_struct: core::mem::MaybeUninit<SP_Device> = core::mem::MaybeUninit::uninit();
    device.struct_size = offset_of!(SP_Device, pci_bus_id)
        + std::mem::size_of_val(&dumb_struct.assume_init().pci_bus_id); // This is a re-impl of the SP_DEVICE_STRUCT_SIZE macro

    let new_device: Box<backend::VultenInstance> = Box::new(backend::VultenInstance::new(Some(
        (*params).ordinal as usize,
    )));
    let device_ptr = Box::leak(new_device) as *mut backend::VultenInstance;
    device.device_handle = device_ptr as *mut c_void;
    device.ordinal = (*params).ordinal;
    device.hardware_name = (*device_ptr).get_device_name();

    GLOBAL_INSTANCES.write().unwrap().insert(
        device.ordinal as usize,
        device.device_handle as *mut backend::VultenInstance,
    );

    log_init!(
        "device: {:p} ordinal: {:?}",
        device.device_handle,
        (*params).ordinal
    );
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_destroy_device(
    _platform: *const SP_Platform,
    device: *mut SP_Device,
) {
    log_init!("");

    let _: Box<backend::VultenInstance> =
        Box::from_raw((*device).device_handle as *mut backend::VultenInstance);
    (*device).device_handle = std::ptr::null_mut::<c_void>(); // free what this points to.
    (*device).ordinal = -1;
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_create_device_fns(
    _platform: *const SP_Platform,
    params: *mut SE_CreateDeviceFnsParams,
    _status: *mut TF_Status,
) {
    log_init!("");

    let dumb_struct: core::mem::MaybeUninit<SP_DeviceFns> = core::mem::MaybeUninit::uninit();
    (*(*params).device_fns).struct_size = offset_of!(SP_DeviceFns, get_gflops)
        + std::mem::size_of_val(&dumb_struct.assume_init().get_gflops); // This is a re-impl of the SP_DEVICE_FNS_STRUCT_SIZE macro
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_destroy_device_fns(
    _platform: *const SP_Platform,
    _device_fns: *mut SP_DeviceFns,
) {
    log_init!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_synchronize_all_activity(
    _device: *const SP_Device,
    status: *mut TF_Status,
) {
    log_init!("");
    TF_SetStatus(status, TSL_Code_TSL_OK, c"".as_ptr());
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_host_callback(
    _device: *const SP_Device,
    _stream: SP_Stream,
    _callback_fn: SE_StatusCallbackFn,
    _callback_arg: *mut c_void,
) -> TF_Bool {
    log_init!("");
    true as TF_Bool
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_create_stream_executor(
    _platform: *const SP_Platform,
    params: *mut SE_CreateStreamExecutorParams,
    _status: *mut TF_Status,
) {
    log_init!("");

    let stream_executor = &mut *(*params).stream_executor;

    let dumb_struct: core::mem::MaybeUninit<SP_StreamExecutor> = core::mem::MaybeUninit::uninit();
    stream_executor.struct_size = offset_of!(SP_StreamExecutor, host_callback)
        + std::mem::size_of_val(&dumb_struct.assume_init().host_callback); // This is a re-impl of the SP_STREAMEXECUTOR_STRUCT_SIZE macro
    stream_executor.synchronize_all_activity = Some(plugin_synchronize_all_activity);
    stream_executor.host_callback = Some(plugin_host_callback);

    stream_executor.allocate = Some(allocaters::plugin_allocate);
    stream_executor.deallocate = Some(allocaters::plugin_deallocate);
    stream_executor.get_allocator_stats = Some(allocaters::plugin_get_allocator_stats);
    stream_executor.host_memory_allocate = Some(allocaters::plugin_host_memory_allocate);
    stream_executor.host_memory_deallocate = Some(allocaters::plugin_host_memory_deallocate);
    stream_executor.device_memory_usage = Some(allocaters::plugin_device_memory_usage);

    stream_executor.create_stream = Some(stream::plugin_create_stream);
    stream_executor.destroy_stream = Some(stream::plugin_destroy_stream);
    stream_executor.create_stream_dependency = Some(stream::plugin_create_stream_dependency);
    stream_executor.get_stream_status = Some(stream::plugin_get_stream_status);
    stream_executor.block_host_until_done = Some(stream::plugin_block_host_until_done);

    stream_executor.create_event = Some(event::plugin_create_event);
    stream_executor.destroy_event = Some(event::plugin_destroy_event);
    stream_executor.get_event_status = Some(event::plugin_get_event_status);
    stream_executor.record_event = Some(event::plugin_record_event);
    stream_executor.wait_for_event = Some(event::plugin_wait_for_event);
    stream_executor.block_host_for_event = Some(event::plugin_block_host_for_event);

    stream_executor.memcpy_dtoh = Some(memory::plugin_memcpy_dtoh);
    stream_executor.memcpy_htod = Some(memory::plugin_memcpy_htod);
    stream_executor.memcpy_dtod = Some(memory::plugin_memcpy_dtod);
    stream_executor.sync_memcpy_dtoh = Some(memory::plugin_sync_memcpy_dtoh);
    stream_executor.sync_memcpy_htod = Some(memory::plugin_sync_memcpy_htod);
    stream_executor.sync_memcpy_dtod = Some(memory::plugin_sync_memcpy_dtod);
    stream_executor.mem_zero = Some(memory::pulgin_mem_zero);
    stream_executor.memset = Some(memory::pulgin_memset);
    stream_executor.memset32 = Some(memory::pulgin_memset32);
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_destroy_stream_executor(
    _platform: *const SP_Platform,
    _stream_executor: *mut SP_StreamExecutor,
) {
    log_init!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_destroy_platform(_platform: *mut SP_Platform) {
    log_init!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_destroy_platform_fns(_platform_fns: *mut SP_PlatformFns) {
    log_init!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn SE_InitPluginFns(
    params: *mut SE_PlatformRegistrationParams,
    _status: *mut TF_Status,
) {
    log_init!("");

    let platform = &mut *(*params).platform_fns;
    platform.get_device_count = Some(plug_get_device_count);
    platform.create_device = Some(plugin_create_device);
    platform.destroy_device = Some(plugin_destroy_device);
    platform.create_device_fns = Some(plugin_create_device_fns);
    platform.destroy_device_fns = Some(plugin_destroy_device_fns);
    platform.create_stream_executor = Some(plugin_create_stream_executor);
    platform.destroy_stream_executor = Some(plugin_destroy_stream_executor);

    (*params).destroy_platform = Some(plugin_destroy_platform);
    (*params).destroy_platform_fns = Some(plugin_destroy_platform_fns);
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn SE_InitPlugin(
    params: *mut SE_PlatformRegistrationParams,
    status: *mut TF_Status,
) {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(true)
        .init();

    info!("Plugin Init");

    let platform = &mut *(*params).platform;

    let dumb_struct: core::mem::MaybeUninit<SP_Platform> = core::mem::MaybeUninit::uninit();
    platform.struct_size = offset_of!(SP_Platform, force_memory_growth)
        + std::mem::size_of_val(&dumb_struct.assume_init().force_memory_growth); // This is a re-impl of the SP_PLATFORM_STRUCT_SIZE macro
    platform.name = DEVICE_NAME;
    platform.type_ = DEVICE_TYPE;
    platform.force_memory_growth = false as TF_Bool;
    platform.use_bfc_allocator = false as TF_Bool;

    SE_InitPluginFns(params, status);
}
