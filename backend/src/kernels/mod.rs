use std::ops::Range;

use crate::va::VaAddress;

pub mod assign_add_sub_variable;
pub mod binary;
pub mod matmul;
pub mod reduce;
pub mod relu;
pub mod ssxent;
pub mod unary;

#[derive(Debug, Clone, Copy)]
pub struct KernelInput<'a> {
    pub addr: VaAddress,
    pub dims: &'a [i64],
}

#[derive(Debug, Default)]
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
            chunks.push(i * chunk_size..i * chunk_size + chunk_size.min(ammount_left));
            ammount_left -= chunk_size;
        }

        chunks
    }
}
