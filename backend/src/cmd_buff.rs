use ash::{
    prelude::VkResult,
    vk::{
        self, Buffer, BufferCopy, BufferMemoryBarrier, CommandBuffer, CommandBufferUsageFlags,
        DependencyFlags, DescriptorSet, ImageMemoryBarrier, MemoryBarrier, PipelineBindPoint,
        PipelineLayout, PipelineStageFlags, ShaderStageFlags,
    },
};
pub use ash::{Device, Instance};
use std::sync::{Arc, MutexGuard};

use crate::pipeline::VultenPipeline;

pub struct CommandBufferBuilder<'a> {
    cmd_buff: CommandBuffer,
    device: &'a Device,
    has_began: bool,
    has_ended: bool,
}

impl CommandBufferBuilder<'_> {
    pub fn new(cmd_buff: CommandBuffer, device: &Device) -> CommandBufferBuilder {
        CommandBufferBuilder {
            cmd_buff,
            device,
            has_began: false,
            has_ended: false,
        }
    }

    pub fn begin(mut self) -> Self {
        let bgn_info = vk::CommandBufferBeginInfo::builder()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe { self.device.begin_command_buffer(self.cmd_buff, &bgn_info) }.unwrap();
        self.has_began = true;
        self
    }

    pub fn end(mut self) -> Self {
        unsafe { self.device.end_command_buffer(self.cmd_buff) }.unwrap();
        self.has_ended = true;
        self
    }

    pub fn copy_buffer(self, src_buffer: Buffer, dst_buffer: Buffer, cpy_info: BufferCopy) -> Self {
        unsafe {
            self.device
                .cmd_copy_buffer(self.cmd_buff, src_buffer, dst_buffer, &[cpy_info])
        };
        self
    }

    pub fn pipeline_barrier(
        self,
        src_mask: PipelineStageFlags,
        dst_mask: PipelineStageFlags,
        dependency_flags: DependencyFlags,
        memory_barriers: &[MemoryBarrier],
        buffer_memory_barriers: &[BufferMemoryBarrier],
        image_barriers: &[ImageMemoryBarrier],
    ) -> Self {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.cmd_buff,
                src_mask,
                dst_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_barriers,
            )
        };
        self
    }

    pub fn bind_pipeline(
        self,
        pipeline_bind_point: PipelineBindPoint,
        pipeline: Arc<VultenPipeline>,
    ) -> Self {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.cmd_buff, pipeline_bind_point, pipeline.pipeline);
        }
        self
    }

    pub fn bind_descriptor_sets(
        self,
        pipeline_bind_point: PipelineBindPoint,
        layout: PipelineLayout,
        first_set: u32,
        sets: &[DescriptorSet],
        dyn_offsets: &[u32],
    ) -> Self {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.cmd_buff,
                pipeline_bind_point,
                layout,
                first_set,
                sets,
                dyn_offsets,
            );
        }
        self
    }

    pub fn push_constants(
        self,
        layout: PipelineLayout,
        stage_flags: ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) -> Self {
        unsafe {
            self.device
                .cmd_push_constants(self.cmd_buff, layout, stage_flags, offset, data);
        }
        self
    }

    pub fn dispatch(self, x: u32, y: u32, z: u32) -> Self {
        unsafe {
            self.device.cmd_dispatch(self.cmd_buff, x, y, z);
        }
        self
    }

    pub fn fill_buffer(self, buffer: Buffer, offset: u64, size: u64, data: u32) -> Self {
        unsafe {
            self.device
                .cmd_fill_buffer(self.cmd_buff, buffer, offset, size, data);
        }
        self
    }

    //events

    pub fn build(self) -> Result<(), &'static str> {
        if self.has_began && self.has_ended {
            Ok(())
        } else {
            Err("Command buffers must have begin() and end() to be valid")
        }
    }
}

impl super::VultenInstance {
    pub fn create_cmd_buffers(
        &self,
        count: u32,
        queue: &MutexGuard<super::queue::VultenQueue>,
    ) -> VkResult<Vec<CommandBuffer>> {
        let cmd_info: vk::CommandBufferAllocateInfo = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(count)
            .command_pool(queue.pool)
            .build();

        unsafe { self.device.allocate_command_buffers(&cmd_info) }
    }

    pub fn free_cmd_buffers(
        &self,
        queue: &MutexGuard<super::queue::VultenQueue>,
        cmd_buffers: Vec<CommandBuffer>,
    ) {
        unsafe {
            self.device.free_command_buffers(queue.pool, &cmd_buffers);
        }
    }
}
