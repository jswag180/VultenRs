use std::{ops::Range, sync::Arc};

use ash::vk;

use crate::{memory::VultenBuffer, va::VaAddress, GOLBAL_DEVICE_VA};

pub mod assign_add_sub_variable;
pub mod binary;
pub mod conv2d;
pub mod matmul;
pub mod reduce;
pub mod relu;
pub mod ssxent;
pub mod unary;

// #[derive(Debug, Clone, Copy)]
// pub struct KernelInput<'a> {
//     pub addr: VaAddress,
//     pub dims: &'a [i64],
// }

#[derive(Clone)]
pub enum KernelBuff<'a> {
    Addr(VaAddress),
    Buff(Arc<VultenBuffer<'a>>),
}

impl KernelBuff<'_> {
    pub fn get_descriptor_info(&self) -> Result<[vk::DescriptorBufferInfo; 1], &'static str> {
        match self {
            Self::Addr(addr) => {
                let alloc = GOLBAL_DEVICE_VA.find_va(*addr)?;

                Ok([vk::DescriptorBufferInfo::default()
                    .buffer(alloc.0.obj.vk_buffer)
                    .range(alloc.0.size - alloc.1)
                    .offset(alloc.1)])
            }
            Self::Buff(buff) => Ok([vk::DescriptorBufferInfo::default()
                .buffer(buff.vk_buffer)
                .range(buff.size)
                .offset(0)]),
        }
    }

    pub fn get_buffer(&self) -> Result<(Arc<VultenBuffer<'_>>, u64), &'static str> {
        match self {
            Self::Addr(addr) => {
                let alloc = GOLBAL_DEVICE_VA.find_va(*addr)?;
                Ok((alloc.0.obj, alloc.1))
            }
            Self::Buff(buff) => Ok((buff.clone(), 0)),
        }
    }
}

impl From<VaAddress> for KernelBuff<'_> {
    fn from(value: VaAddress) -> Self {
        Self::Addr(value)
    }
}

#[derive(Clone)]
pub struct KernelInput<'a> {
    pub buff: KernelBuff<'a>,
    pub dims: &'a [i64],
}

#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Copy)]
pub enum ChannelFormat {
    #[default]
    NHWC,
    NCHW,
}

impl TryFrom<&str> for ChannelFormat {
    type Error = &'static str;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "NHWC" => Ok(Self::NHWC),
            "NCHW" => Ok(Self::NCHW),
            _ => Err("Invalid format!"),
        }
    }
}

trait Chunkable<T> {
    fn as_chunks<'a>(&self, chunk_size: T) -> Vec<Range<T>>;
}

impl Chunkable<i64> for Range<i64> {
    fn as_chunks<'a>(&self, chunk_size: i64) -> Vec<Range<i64>> {
        let total_chunks = ((self.end - self.start) as f32 / chunk_size as f32).ceil() as i64;
        let mut chunks: Vec<Range<i64>> = Vec::with_capacity(total_chunks as usize);

        let mut ammount_left = self.end;

        for i in 0..total_chunks {
            chunks.push(
                (i * chunk_size) + self.start
                    ..i * chunk_size + chunk_size.min(ammount_left) + self.start,
            );
            ammount_left -= chunk_size;
        }

        chunks
    }
}
