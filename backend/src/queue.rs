use ash::{
    prelude::VkResult,
    vk::{self, Fence, SubmitInfo},
};
pub use ash::{Device, Instance};
use std::sync::MutexGuard;

//#[derive(Debug)]
//pub struct VultenQueueFlags(vk::QueueFlags);
pub type VultenQueueFlags = vk::QueueFlags;

pub struct CommandSet {}

#[derive(Debug)]
pub struct VultenQueue {
    pub queue: vk::Queue,
    pub capability: VultenQueueFlags,
    pub pool: vk::CommandPool,
}

impl VultenQueue {
    pub fn new(queue: vk::Queue, capability: VultenQueueFlags, pool: vk::CommandPool) -> Self {
        Self {
            queue,
            capability,
            pool,
        }
    }
}

impl super::VultenInstance {
    pub fn get_queue(&self, flags: VultenQueueFlags) -> MutexGuard<VultenQueue> {
        loop {
            for q in 0..self.queues.len() {
                let locked_q = self.queues[q].try_lock();
                if let Ok(i) = locked_q {
                    if i.capability.contains(flags) {
                        return i;
                    } else {
                        continue;
                    }
                }
            }
        }
    }

    pub fn submit_queue(
        &self,
        queue: &MutexGuard<VultenQueue>,
        sub_info: &[SubmitInfo],
        fence: Fence,
    ) -> VkResult<()> {
        unsafe { self.device.queue_submit(queue.queue, sub_info, fence) }
    }
}
