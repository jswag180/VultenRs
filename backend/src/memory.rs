use ash::prelude::VkResult;
use ash::vk::{self, Buffer, BufferUsageFlags, DeviceSize, QueueFlags};
pub use ash::{Device, Instance};
use std::any::type_name;
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::MutexGuard;
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator};

use crate::queue::VultenQueueFlags;
use crate::va::VaAddress;
use crate::GOLBAL_DEVICE_VA;

pub type VultenCpyInfo = vk::BufferCopy;

#[derive(Debug)]
pub enum MemoryError {
    NotMappable,
    FaildToMap,
}

#[derive(Debug)]
pub enum TransferError {
    QueueIsNotTransfer,
    SizeMisMatch,
    NonDeviceBuffer,
}

#[derive(Debug, PartialEq, Clone)]
pub enum VultenBufferType {
    Device,
    Host,
    Staging,
    Uiniform,
    //Img
}

pub struct VultenBuffer<'a> {
    pub buff_type: VultenBufferType,
    pub vk_buffer: Buffer,
    pub allocation: Allocation,
    pub size: DeviceSize,
    pub allocator: &'a Allocator,
}

impl Drop for VultenBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.vk_buffer, &mut self.allocation)
        };
    }
}

impl VultenBuffer<'_> {
    pub fn get_mapped_ptr(&self) -> Result<*mut c_void, MemoryError> {
        match self.buff_type {
            VultenBufferType::Host | VultenBufferType::Staging | VultenBufferType::Uiniform => {
                Ok(self
                    .allocator
                    .get_allocation_info(&self.allocation)
                    .mapped_data)
            }
            _ => Err(MemoryError::NotMappable),
        }
    }

    pub fn get_descriptor_info(
        &self,
        size: Option<DeviceSize>,
        offset: Option<DeviceSize>,
    ) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.vk_buffer)
            .range(size.unwrap_or(self.size))
            .offset(offset.unwrap_or(0))
    }
}

impl super::VultenInstance {
    pub fn get_mem_stats(&self) -> (i64, i64) {
        let budgets = self.allocator.get_heap_budgets().unwrap();
        let mem_props = unsafe { self.allocator.get_memory_properties() };
        for i in 0..mem_props.memory_heap_count {
            if mem_props.memory_heaps[i as usize]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            {
                return (
                    (budgets[i as usize].budget - budgets[i as usize].usage) as i64,
                    budgets[i as usize].budget as i64,
                );
            }
        }

        (0, 0)
    }

    pub fn create_buffer(
        &self,
        buff_type: VultenBufferType,
        size: DeviceSize,
        transfer_src: bool,
        transfer_dst: bool,
    ) -> VultenBuffer {
        let mut create_flags = AllocationCreateFlags::empty();
        match buff_type {
            VultenBufferType::Host => {
                create_flags |=
                    AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_RANDOM
            }
            VultenBufferType::Staging | VultenBufferType::Uiniform => {
                create_flags |= AllocationCreateFlags::MAPPED
                    | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
            }
            _ => (),
        }

        let create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            flags: create_flags,
            ..Default::default()
        };

        let mut usage = BufferUsageFlags::default();
        if transfer_src {
            usage |= BufferUsageFlags::TRANSFER_SRC;
        }
        if transfer_dst {
            usage |= BufferUsageFlags::TRANSFER_DST;
        }
        match buff_type {
            VultenBufferType::Uiniform => usage |= BufferUsageFlags::UNIFORM_BUFFER,
            _ => usage |= BufferUsageFlags::STORAGE_BUFFER,
        }

        let (buffer, allocation) = unsafe {
            self.allocator.create_buffer(
                &ash::vk::BufferCreateInfo::default().size(size).usage(usage),
                &create_info,
            )
        }
        .unwrap();

        VultenBuffer {
            buff_type,
            vk_buffer: buffer,
            allocation,
            size,
            allocator: &self.allocator,
        }
    }

    /// Upload a host byte slice to a device buffer.
    /// # Arguments
    ///
    /// * 'data' - byte slice of data to upload.
    /// * 'device_buff' - buffer of type Device to upload to.
    /// * 'offset' - offset in bytes into the 'device_buff' to copy into.
    /// * 'queue' - if it should get its own queue or use supplyed one.
    ///
    /// # Considerations
    /// This blocks on the compleation of the transfer.
    /// If the supplyed queue is not transfer able it will return a 'QueueIsNotTransfer'
    pub fn upload_to_device_buff<T>(
        &self,
        data: &[T],
        device_buff: &VultenBuffer,
        offset: DeviceSize,
        queue: Option<&MutexGuard<super::queue::VultenQueue>>,
    ) -> Result<(), TransferError> {
        if device_buff.buff_type != VultenBufferType::Device {
            return Err(TransferError::NonDeviceBuffer);
        }

        let qu: MutexGuard<super::queue::VultenQueue>;
        let q = match queue {
            Some(i) => {
                if !i.capability.contains(QueueFlags::TRANSFER) {
                    return Err(TransferError::QueueIsNotTransfer);
                }
                i
            }
            None => {
                qu = self.get_queue(QueueFlags::TRANSFER);
                &qu
            }
        };

        if data.len() as u64 > (device_buff.size + offset) {
            return Err(TransferError::SizeMisMatch);
        }

        let buffer_size = core::mem::size_of_val(data);
        let staging =
            self.create_buffer(VultenBufferType::Staging, buffer_size as u64, true, false);
        let staging_ptr = staging.get_mapped_ptr().unwrap() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, staging_ptr, buffer_size)
        };

        let cmd_buff = self
            .create_cmd_buffers(1, q)
            .expect("Error could not allocate cmd buffer");

        let cpy_info = vk::BufferCopy::default()
            .size(buffer_size as u64)
            .dst_offset(offset);

        super::cmd_buff::CommandBufferBuilder::new(cmd_buff[0], &self.device)
            .begin()
            .copy_buffer(staging.vk_buffer, device_buff.vk_buffer, cpy_info)
            .end()
            .build()
            .unwrap();

        unsafe {
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buff);
            self.device
                .queue_submit(q.queue, &[submit_info], vk::Fence::null())
                .expect("Error failed to submit cmd buffer!");
            self.device
                .queue_wait_idle(q.queue)
                .expect("Error timed out wait for transfer!");
        };

        self.free_cmd_buffers(q, cmd_buff);

        Ok(())
    }

    pub fn blocking_cpy(&self, src: vk::Buffer, dst: vk::Buffer, cpy_info: VultenCpyInfo) {
        let q = self.get_queue(VultenQueueFlags::TRANSFER);

        let cmd_buff = self
            .create_cmd_buffers(1, &q)
            .expect("Error could not allocate cmd buffer");

        super::cmd_buff::CommandBufferBuilder::new(cmd_buff[0], &self.device)
            .begin()
            .copy_buffer(src, dst, cpy_info)
            .end()
            .build()
            .unwrap();

        unsafe {
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buff);
            self.device
                .queue_submit(q.queue, &[submit_info], vk::Fence::null())
                .expect("Error failed to submit cmd buffer!");
            self.device
                .queue_wait_idle(q.queue)
                .expect("Error timed out wait for transfer!");
        };

        self.free_cmd_buffers(&q, cmd_buff);
    }

    pub fn get_descriptor_info_va(
        addr: VaAddress,
    ) -> Result<([vk::DescriptorBufferInfo; 1], Buffer), &'static str> {
        let alloc = unsafe { GOLBAL_DEVICE_VA.find_va(addr)? };

        Ok((
            [vk::DescriptorBufferInfo::default()
                .buffer(alloc.0.obj.vk_buffer)
                .range(alloc.0.size - alloc.1)
                .offset(alloc.1)],
            alloc.0.obj.vk_buffer,
        ))
    }

    pub fn fill_buffer(
        &self,
        buff: &VultenBuffer,
        size: u64,
        offset: u64,
        data: u32,
    ) -> VkResult<()> {
        let q = self.get_queue(VultenQueueFlags::TRANSFER);

        let cmd_buff = self.create_cmd_buffers(1, &q)?;

        super::cmd_buff::CommandBufferBuilder::new(cmd_buff[0], &self.device)
            .begin()
            .fill_buffer(buff.vk_buffer, offset, size, data)
            .end()
            .build()
            .unwrap();

        unsafe {
            let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buff);
            self.device
                .queue_submit(q.queue, &[submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(q.queue)?;
        };

        self.free_cmd_buffers(&q, cmd_buff);
        Ok(())
    }

    pub unsafe fn dump_buffer<T, F>(
        &self,
        buff: &VultenBuffer,
        range_bytes: u64,
        offset: u64,
        func: F,
    ) -> Result<(), String>
    where
        F: Fn(&[T]),
        T: Sized,
    {
        let num_vals = (range_bytes / size_of::<T>() as u64) as usize;
        if num_vals < 1 {
            return Err(format!(
                "range_bytes: {:?} is greater then size of {:}: {:?}",
                range_bytes,
                type_name::<T>(),
                size_of::<T>()
            ));
        }

        let buffer_to_dump = if buff.buff_type == VultenBufferType::Host {
            buff
        } else {
            let host_buff =
                Rc::new(self.create_buffer(VultenBufferType::Host, range_bytes, false, true));
            let cpy_info = VultenCpyInfo::default()
                .size(range_bytes)
                .src_offset(offset)
                .dst_offset(0);
            self.blocking_cpy(buff.vk_buffer, host_buff.vk_buffer, cpy_info);
            &host_buff.clone()
        };

        let slice: &[T] = std::slice::from_raw_parts(
            buffer_to_dump.get_mapped_ptr().unwrap() as *mut T,
            num_vals,
        );
        func(slice);
        Ok(())
    }
}
