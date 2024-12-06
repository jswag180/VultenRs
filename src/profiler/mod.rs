use std::{
    collections::VecDeque,
    mem::offset_of,
    sync::{atomic::AtomicBool, LazyLock, Mutex, RwLock},
    time::Instant,
};

use crate::{log_prof, DEVICE_TYPE};
use profile_data::ProfileData;
use protobuf::Message;
use tensorflow_pluggable_device_sys::{
    TF_ProfilerRegistrationParams, TF_Status, TP_Profiler, TP_ProfilerFns,
};

pub mod profile_data;
pub mod xplane_utils;

pub static START_TIME: RwLock<Option<Instant>> = RwLock::new(None);
pub static IS_PROFILING: AtomicBool = AtomicBool::new(false);
pub static PROFILER_DATA: LazyLock<Mutex<VecDeque<ProfileData>>> = LazyLock::new(Mutex::default);

static SPACES: Mutex<Option<Vec<u8>>> = Mutex::new(None);

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_profiler_start(
    profiler: *const TP_Profiler,
    status: *mut TF_Status,
) {
    PROFILER_DATA.lock().unwrap().clear();
    *SPACES.lock().unwrap() = None;

    let mut start_time = START_TIME.write().unwrap();
    *start_time = Some(Instant::now());

    IS_PROFILING.store(true, std::sync::atomic::Ordering::SeqCst);

    log_prof!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_profiler_stop(
    profiler: *const TP_Profiler,
    status: *mut TF_Status,
) {
    IS_PROFILING.store(false, std::sync::atomic::Ordering::SeqCst);
    log_prof!("");
}

/// This is called 2 times the first time is just to figure out the size of buffer needed
#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_profiler_collect_data_xspace(
    profiler: *const TP_Profiler,
    buffer: *mut u8,
    size_in_bytes: *mut usize,
    status: *mut TF_Status,
) {
    log_prof!("size_in_bytes: {:}", *size_in_bytes);

    if buffer.is_null() {
        let mut space_cache = SPACES.lock().unwrap();
        if space_cache.is_none() {
            let space = xplane_utils::generate_xspace();
            let space_bytes = space.write_to_bytes().unwrap();

            *size_in_bytes = space_bytes.len();
            *space_cache = Some(space_bytes);
        }
        return;
    } else {
        let space_cache = SPACES.lock().unwrap();
        if let Some(space_bytes) = space_cache.as_ref() {
            assert!(space_bytes.len() == *size_in_bytes);

            libc::memcpy(buffer as _, space_bytes.as_ptr() as _, *size_in_bytes);
        }
        return;
    }
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_profiler_destroy_profiler(profiler: *mut TP_Profiler) {
    log_prof!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_profiler_destroy_profiler_fns(profiler: *mut TP_ProfilerFns) {
    log_prof!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn TF_InitProfiler(
    params: *mut TF_ProfilerRegistrationParams,
    _status: *mut TF_Status,
) {
    log_prof!("Init Profiler");

    let dumb_struct: core::mem::MaybeUninit<TF_ProfilerRegistrationParams> =
        core::mem::MaybeUninit::uninit();
    (*params).struct_size = offset_of!(TF_ProfilerRegistrationParams, destroy_profiler_fns)
        + std::mem::size_of_val(&dumb_struct.assume_init().destroy_profiler_fns); // This is a re-impl of the TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE
    let dumb_struct: core::mem::MaybeUninit<TP_ProfilerFns> = core::mem::MaybeUninit::uninit();
    (*(*params).profiler_fns).struct_size = offset_of!(TP_ProfilerFns, collect_data_xspace)
        + std::mem::size_of_val(&dumb_struct.assume_init().collect_data_xspace); // This is a re-impl of the TP_PROFILER_FNS_STRUCT_SIZE

    (*(*params).profiler).device_type = DEVICE_TYPE;

    (*(*params).profiler_fns).start = Some(plugin_profiler_start);
    (*(*params).profiler_fns).stop = Some(plugin_profiler_stop);
    (*(*params).profiler_fns).collect_data_xspace = Some(plugin_profiler_collect_data_xspace);

    (*params).destroy_profiler = Some(plugin_profiler_destroy_profiler);
    (*params).destroy_profiler_fns = Some(plugin_profiler_destroy_profiler_fns);
}
