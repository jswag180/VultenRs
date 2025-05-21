use ash::vk::{
    self, PushConstantRange, ShaderStageFlags, SpecializationInfo, SpecializationMapEntry,
};
use matmul_inline_transpose::MatMulKernelInline;
use matmul_non_inline::MatMulKernelNonInline;
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    cmd_buff::CommandBufferBuilder,
    compiler,
    descriptor::VultenDescriptor,
    pipeline::{PipelineSpec, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

use super::KernelBuff;

pub const MATMUL_SOURCE: &str = include_str!("matmul.comp");

pub const MAX_BLOCK_SIZE: i64 = 16;
const SMALL_CUTOFF: i64 = 32 * 32;
pub const BROADCAST_NONE: u32 = 0;
pub const BROADCAST_A: u32 = 1;
pub const BROADCAST_B: u32 = 2;

pub mod matmul_inline_transpose;
pub mod matmul_non_inline;
pub mod transpose;

#[cfg(test)]
mod matmul_test;

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct MatmulPipelineSpec {
    local_x: u32,
    block_size_x: u32,
    block_size_y: u32,
    bk_cont: u32,
    a_x: u32,
    a_y: u32,
    b_x: u32,
    b_y: u32,
    inline_trans_a: bool,
    inline_trans_b: bool,
    bk_num_y: u32,
    broadcast: u32,
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes, Default)]
#[repr(C, packed)]
pub struct MatmulPushConst {
    start_x: u32,
    stop_x: u32,
    offset: u32,
}

impl PushConstSpec for MatmulPushConst {
    fn get_ranges() -> &'static [PushConstantRange] {
        &[PushConstantRange {
            offset: 0,
            stage_flags: ShaderStageFlags::COMPUTE,
            size: std::mem::size_of::<Self>() as u32,
        }]
    }

    #[inline]
    fn get_slice(&self) -> &[u8] {
        let slice: &[u8; 12] = zerocopy::transmute_ref!(self);

        slice
    }
}

impl PipelineSpec for MatmulPipelineSpec {
    type PushConst = MatmulPushConst;

    fn get_shader(&self) -> Vec<u32> {
        let mut compiler: compiler::ShaderCompiler = compiler::ShaderCompiler::new(MATMUL_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.add_define(
            "MAX_BLOCK_SIZE".into(),
            Some(self.block_size_x.max(self.block_size_y).to_string()),
        );

        compiler.compile().unwrap()
    }

    fn get_spec_info(&self) -> (Box<[SpecializationMapEntry]>, Vec<u8>) {
        //offset needs to be the offset in the spec_buffer vec not the struct
        let spec_entrys = [
            SpecializationMapEntry {
                constant_id: 0,
                offset: 0,
                size: std::mem::size_of_val(&self.local_x),
            },
            SpecializationMapEntry {
                constant_id: 1,
                offset: 4,
                size: std::mem::size_of_val(&self.block_size_x),
            },
            SpecializationMapEntry {
                constant_id: 2,
                offset: 8,
                size: std::mem::size_of_val(&self.block_size_y),
            },
            SpecializationMapEntry {
                constant_id: 3,
                offset: 12,
                size: std::mem::size_of_val(&self.bk_cont),
            },
            SpecializationMapEntry {
                constant_id: 4,
                offset: 16,
                size: std::mem::size_of_val(&self.a_x),
            },
            SpecializationMapEntry {
                constant_id: 5,
                offset: 20,
                size: std::mem::size_of_val(&self.a_y),
            },
            SpecializationMapEntry {
                constant_id: 6,
                offset: 24,
                size: std::mem::size_of_val(&self.b_x),
            },
            SpecializationMapEntry {
                constant_id: 7,
                offset: 28,
                size: std::mem::size_of_val(&self.b_y),
            },
            SpecializationMapEntry {
                constant_id: 8,
                offset: 32,
                size: std::mem::size_of::<vk::Bool32>(),
            },
            SpecializationMapEntry {
                constant_id: 9,
                offset: 36,
                size: std::mem::size_of::<vk::Bool32>(),
            },
            SpecializationMapEntry {
                constant_id: 10,
                offset: 40,
                size: std::mem::size_of_val(&self.bk_num_y),
            },
            SpecializationMapEntry {
                constant_id: 11,
                offset: 44,
                size: std::mem::size_of_val(&self.broadcast),
            },
        ];

        let mut spec_buffer: Vec<u8> = Vec::new();
        let local_x = self.local_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&local_x);
        let block_size_x = self.block_size_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&block_size_x);
        let block_size_y = self.block_size_y.to_ne_bytes();
        spec_buffer.extend_from_slice(&block_size_y);
        let bk_cont = self.bk_cont.to_ne_bytes();
        spec_buffer.extend_from_slice(&bk_cont);
        let a_x = self.a_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&a_x);
        let a_y = self.a_y.to_ne_bytes();
        spec_buffer.extend_from_slice(&a_y);
        let b_x = self.b_x.to_ne_bytes();
        spec_buffer.extend_from_slice(&b_x);
        let b_y = self.b_y.to_ne_bytes();
        spec_buffer.extend_from_slice(&b_y);
        let inline_trans_a = (self.inline_trans_a as vk::Bool32).to_ne_bytes();
        spec_buffer.extend_from_slice(&inline_trans_a);
        let inline_trans_b = (self.inline_trans_b as vk::Bool32).to_ne_bytes();
        spec_buffer.extend_from_slice(&inline_trans_b);
        let bk_num_y = self.bk_num_y.to_ne_bytes();
        spec_buffer.extend_from_slice(&bk_num_y);
        let broadcast = self.broadcast.to_ne_bytes();
        spec_buffer.extend_from_slice(&broadcast);

        debug_assert!(spec_buffer.len() <= spec_entrys.iter().fold(0, |acc, x| acc + x.size));

        (Box::new(spec_entrys), spec_buffer)
    }

    fn build_pipeline(&self, inst: &VultenInstance) -> Arc<VultenPipeline> {
        let desc_types: Vec<vk::DescriptorType> = vec![vk::DescriptorType::STORAGE_BUFFER; 3];
        let shader = self.get_shader();
        let spec_info = self.get_spec_info();

        let pipe = inst
            .create_compute_pipeline(
                desc_types,
                &shader,
                Some(
                    &SpecializationInfo::default()
                        .map_entries(&spec_info.0)
                        .data(&spec_info.1),
                ),
                Self::PushConst::get_ranges(),
            )
            .unwrap();

        Arc::new(pipe)
    }
}

pub fn get_block_dims(mat_a: (i64, i64), mat_b: (i64, i64)) -> (u32, u32) {
    let mut block_size: (i64, i64) = (0, 0);

    let total_a = mat_a.0 * mat_a.1;
    let total_b = mat_b.0 * mat_b.1;
    let max_block_size = if total_a <= SMALL_CUTOFF || total_b <= SMALL_CUTOFF {
        4
    } else {
        MAX_BLOCK_SIZE
    };

    for i in (0..max_block_size).rev() {
        if mat_b.0 % i == 0 && mat_a.0 % i == 0 {
            block_size.0 = i;
            break;
        }
    }

    for i in (0..max_block_size).rev() {
        if mat_b.1 % i == 0 && mat_a.1 % i == 0 {
            block_size.1 = i;
            break;
        }
    }

    (block_size.0 as u32, block_size.1 as u32)
}

pub enum Version {
    Inline,
    NonInline,
}

pub trait MatMulKernelVersion<'a> {
    fn get_pipeline(&mut self) -> Result<Arc<VultenPipeline>, &'static str>;
    fn get_descriptors(
        &mut self,
        pipeline: Arc<VultenPipeline>,
    ) -> Result<Vec<VultenDescriptor<'a>>, &'static str>;
    fn record<'b>(
        &mut self,
        builder: CommandBufferBuilder<'b>,
        pipeline: Arc<VultenPipeline>,
        descriptors: &[VultenDescriptor],
    ) -> Result<CommandBufferBuilder<'b>, &'static str>;
    fn run(&mut self) -> Result<(), &'static str>;
}

pub struct MatMulKernel<'a> {
    inst: &'a VultenInstance,
    d_type: VultenDataType,
    a: Option<KernelBuff<'a>>,
    a_dims: Option<&'a [i64]>,
    a_transpose: bool,
    b: Option<KernelBuff<'a>>,
    b_dims: Option<&'a [i64]>,
    b_transpose: bool,
    output: Option<KernelBuff<'a>>,
    output_dims: Option<&'a [i64]>,
    block_dims: (u32, u32),
    num_blocks: (i64, i64),
}

impl<'a> MatMulKernel<'a> {
    pub fn new(inst: &'a VultenInstance, d_type: VultenDataType) -> Self {
        Self {
            inst,
            d_type,
            a: Default::default(),
            a_dims: Default::default(),
            a_transpose: Default::default(),
            b: Default::default(),
            b_dims: Default::default(),
            b_transpose: Default::default(),
            output: Default::default(),
            output_dims: Default::default(),
            block_dims: Default::default(),
            num_blocks: Default::default(),
        }
    }

    pub fn a(
        mut self,
        buff: KernelBuff<'a>,
        dims: &'a [i64],
        transpose: bool,
    ) -> Result<Self, &'static str> {
        if dims.contains(&0) {
            return Err("Input a has a zero dim!");
        }
        if dims.len() != 2 && dims.len() != 3 {
            return Err("Input a dims len is not 2 or 3!");
        }
        self.a = Some(buff);
        self.a_dims = Some(dims);
        self.a_transpose = transpose;

        Ok(self)
    }

    pub fn b(
        mut self,
        buff: KernelBuff<'a>,
        dims: &'a [i64],
        transpose: bool,
    ) -> Result<Self, &'static str> {
        if dims.contains(&0) {
            return Err("Input b has a zero dim!");
        }
        if dims.len() != 2 && dims.len() != 3 {
            return Err("Input b dims len is not 2 or 3!");
        }
        self.b = Some(buff);
        self.b_dims = Some(dims);
        self.b_transpose = transpose;

        Ok(self)
    }

    pub fn output(mut self, buff: KernelBuff<'a>, dims: &'a [i64]) -> Result<Self, &'static str> {
        if dims.contains(&0) {
            return Err("Output has a zero dim!");
        }
        if dims.len() != 2 && dims.len() != 3 {
            return Err("Ouput dims len is not 2 or 3!");
        }
        self.output = Some(buff);
        self.output_dims = Some(dims);

        Ok(self)
    }

    pub fn build(
        mut self,
        ver_override: Option<Version>,
    ) -> Result<Box<dyn MatMulKernelVersion<'a> + 'a>, &'static str> {
        let a_dims = self.a_dims.as_ref().ok_or("Missing a dims")?;
        let b_dims = self.b_dims.as_ref().ok_or("Missing b dims")?;
        let output_dims = self.output_dims.as_ref().ok_or("Missing output dims")?;

        if a_dims.len() != b_dims.len() || a_dims.len() != output_dims.len() {
            return Err("Dim len mismatch!");
        }

        let offset = a_dims.len() - 2;
        let mat_a_post: (i64, i64) = if self.a_transpose {
            (a_dims[1 + offset], a_dims[offset])
        } else {
            (a_dims[offset], a_dims[1 + offset])
        };
        let mat_b_post: (i64, i64) = if self.b_transpose {
            (b_dims[1 + offset], b_dims[offset])
        } else {
            (b_dims[offset], b_dims[1 + offset])
        };
        self.block_dims = get_block_dims(mat_a_post, mat_b_post);
        let num_blocks_x = (mat_a_post.0 as f32 / self.block_dims.0 as f32).ceil() as i64;
        let num_blocks_y = (mat_b_post.1 as f32 / self.block_dims.1 as f32).ceil() as i64;
        self.num_blocks = (num_blocks_x, num_blocks_y);

        match ver_override {
            Some(ver_override) => match ver_override {
                Version::Inline => Ok(Box::new(MatMulKernelInline::new(self))),
                Version::NonInline => Ok(Box::new(MatMulKernelNonInline::new(self)?)),
            },
            None => {
                if output_dims.iter().product::<i64>() > SMALL_CUTOFF {
                    Ok(Box::new(MatMulKernelNonInline::new(self)?))
                } else {
                    Ok(Box::new(MatMulKernelInline::new(self)))
                }
            }
        }
    }
}
