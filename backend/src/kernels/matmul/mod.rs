use ash::vk::{
    self, PushConstantRange, ShaderStageFlags, SpecializationInfo, SpecializationMapEntry,
};
use shaderc::CompilationArtifact;
use std::sync::Arc;
use zerocopy::AsBytes;

use crate::{
    compiler,
    pipeline::{PipelineSpec, PushConstSpec, VultenPipeline},
    VultenDataType, VultenInstance,
};

pub const MATMUL_SOURCE: &str = include_str!("matmul.comp");

pub const MAX_BLOCK_SIZE: i64 = 16;
const SMALL_CUTOFF: i64 = 32 * 32;

pub mod matmul;
pub mod matmul_inline_transpose;
pub mod transpose;

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
    d_type: VultenDataType,
}

#[derive(Debug, AsBytes)]
#[repr(C, packed)]
pub struct MatmulPushConst {
    start_x: u32,
    stop_x: u32,
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
        let slice: &[u8; 8] = zerocopy::transmute_ref!(self);

        slice
    }
}

impl PipelineSpec for MatmulPipelineSpec {
    type PushConst = MatmulPushConst;

    fn get_shader(&self) -> CompilationArtifact {
        let mut compiler: compiler::ShaderCompiler =
            compiler::ShaderCompiler::new("matmul.comp", MATMUL_SOURCE);
        compiler.add_type_spec(0, self.d_type).unwrap();

        compiler.opts.add_macro_definition(
            "MAX_BLOCK_SIZE",
            Some(
                self.block_size_x
                    .max(self.block_size_y)
                    .to_string()
                    .as_str(),
            ),
        );

        compiler.compile()
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
                shader.as_binary(),
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
