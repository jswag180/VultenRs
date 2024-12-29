use core::{fmt, panic};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use async_task::{Runnable, Task};
use flume::Sender;
use futures::{executor::block_on, Future};
use tensorflow_pluggable_device_sys::{
    SP_Device, SP_Stream, TF_GetStream, TF_OpKernelContext, TF_Status,
};
use tracing::debug;

use crate::{log_stream, ops::kernel_utills::SafeStatus};

pub struct PluginStream<'a> {
    pub inst: *mut backend::VultenInstance,
    pub stream_dependant: Option<*mut PluginStream<'a>>,
    pub hands: Mutex<VecDeque<Task<()>>>,
    sender: Arc<Sender<Runnable>>,
}

impl fmt::Debug for PluginStream<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PluginStream")
            .field("stream_dependant", &self.stream_dependant)
            .field(
                "hands",
                &format!("count: {:?}", self.hands.lock().unwrap().len()),
            )
            .finish()
    }
}

impl PluginStream<'_> {
    #[tracing::instrument]
    fn new<'a>(inst: *mut backend::VultenInstance) -> PluginStream<'a> {
        let (sender, receiver) = flume::unbounded::<Runnable>();
        std::thread::spawn(move || {
            for runnable in receiver {
                runnable.run();
            }
        });

        PluginStream {
            inst,
            stream_dependant: None,
            hands: Mutex::new(VecDeque::new()),
            sender: Arc::new(sender),
        }
    }

    pub fn schedule_future<T: Future<Output = ()> + Send + 'static>(&mut self, future: T) {
        let send_ref = self.sender.clone();
        let schedule = move |runnable| send_ref.send(runnable).unwrap();
        let (runnable, task) = async_task::spawn(future, schedule);
        runnable.schedule();

        self.hands.lock().unwrap().push_back(task);
    }

    #[tracing::instrument]
    pub fn block_on_pending(&mut self) {
        let hands = &mut *self.hands.lock().unwrap();

        if let Some(depnd) = self.stream_dependant {
            unsafe { (*depnd).block_on_pending() };
        }

        while let Some(task) = hands.pop_front() {
            block_on(task);
        }
    }

    /// # Safety
    ///
    /// This function must be called with a valid TF_OpKernelContext ptr.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn from_ctx(ctx: *mut TF_OpKernelContext, status: &SafeStatus) -> &mut Self {
        if ctx.is_null() {
            panic!("Got null for ctx!")
        }

        &mut *(TF_GetStream(ctx, status.status_ptr()) as *mut PluginStream)
    }
}

#[tracing::instrument(skip(stream))]
#[no_mangle]
pub unsafe extern "C" fn plugin_create_stream(
    device: *const SP_Device,
    stream: *mut SP_Stream,
    _status: *mut TF_Status,
) {
    let inst = (*device).device_handle as *mut backend::VultenInstance;
    let new_stream: Box<PluginStream> = Box::new(PluginStream::new(inst));
    let stream_ptr = Box::leak(new_stream) as *mut PluginStream as SP_Stream;
    *stream = stream_ptr;

    log_stream!(stream = ?*stream);
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_destroy_stream(_device: *const SP_Device, stream: SP_Stream) {
    log_stream!("");
    unsafe {
        let _: Box<PluginStream> = Box::from_raw(stream as *mut PluginStream);
    }
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_create_stream_dependency(
    _device: *const SP_Device,
    dependent: SP_Stream,
    other: SP_Stream,
    _status: *mut TF_Status,
) {
    let dependent_stream: *mut PluginStream = dependent as *mut PluginStream;
    let other_stream: *mut PluginStream = other as *mut PluginStream;

    (*dependent_stream).stream_dependant = Some(other_stream);
    log_stream!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_get_stream_status(
    _device: *const SP_Device,
    stream: SP_Stream,
    _status: *mut TF_Status,
) {
    log_stream!("");
}

#[tracing::instrument]
#[no_mangle]
pub unsafe extern "C" fn plugin_block_host_until_done(
    _device: *const SP_Device,
    stream: SP_Stream,
    _status: *mut TF_Status,
) {
    log_stream!("");
}
