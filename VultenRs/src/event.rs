use tensorflow_pluggable_device_sys::{
    SE_EventStatus, SE_EventStatus_SE_EVENT_COMPLETE, SP_Device, SP_Event, SP_Stream, TF_Status,
};

use crate::{log_event, stream::PluginStream};

#[derive(Debug)]
pub enum Event {
    None,
}

//#[repr(C)]
#[derive(Debug)]
pub struct PluginEvent {
    pub id: Event,
    pub hand: Option<usize>,
}

impl PluginEvent {
    #[tracing::instrument]
    fn new() -> Self {
        PluginEvent {
            id: Event::None,
            hand: None,
        }
    }
}

#[tracing::instrument(skip(event))]
#[no_mangle]
pub unsafe extern "C" fn plugin_create_event(
    _device: *const SP_Device,
    event: *mut SP_Event,
    _status: *mut TF_Status,
) {
    let new_event: Box<PluginEvent> = Box::new(PluginEvent::new());
    let event_ptr = Box::leak(new_event) as *mut PluginEvent as SP_Event;
    *event = event_ptr;

    log_event!(event = ?*event);
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_destroy_event(_device: *const SP_Device, event: SP_Event) {
    log_event!("");
    unsafe {
        let _: Box<PluginEvent> = Box::from_raw(event as *mut PluginEvent);
    }
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_get_event_status(
    _device: *const SP_Device,
    event: SP_Event,
) -> SE_EventStatus {
    let plug_event: *mut PluginEvent = event as *mut PluginEvent;
    if let Some(handle) = (*plug_event).hand {
        (*(handle as *mut PluginStream)).block_on_pending();
        log_event!("COMPLETE");
    }

    SE_EventStatus_SE_EVENT_COMPLETE
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_record_event(
    _device: *const SP_Device,
    stream: SP_Stream,
    event: SP_Event,
    _status: *mut TF_Status,
) {
    let plug_stream: *mut super::stream::PluginStream = stream as *mut super::stream::PluginStream;
    let plug_event: *mut PluginEvent = event as *mut PluginEvent;

    (*plug_event).hand = Some(plug_stream.as_ref().unwrap() as *const PluginStream as usize);

    log_event!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_wait_for_event(
    _device: *const SP_Device,
    stream: SP_Stream,
    event: SP_Event,
    _status: *mut TF_Status,
) {
    log_event!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_block_host_for_event(
    _device: *const SP_Device,
    event: SP_Event,
    _status: *mut TF_Status,
) {
    log_event!("");
}
