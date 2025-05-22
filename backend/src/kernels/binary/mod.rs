use std::sync::Arc;

use binary_broad::BinaryKernelBroad;
use binary_no_broad::BinaryKernelNoBroad;
use binary_simple::BinaryKernelSimple;
use shape_helper::BroadcastShapeHelper;

use crate::{
    cmd_buff::CommandBufferBuilder, descriptor::VultenDescriptor, pipeline::VultenPipeline,
    VultenDataType, VultenInstance,
};

use super::KernelBuff;

pub mod binary_broad;
pub mod binary_no_broad;
pub mod binary_simple;
pub mod shape_helper;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum BinaryOp {
    Mul = 0,
    Add = 1,
    Sub = 2,
    Div = 3,
    DivNoNan = 4,
    DivReal = 5,
    Max = 6,
    Min = 7,
    Pow = 8,
    SqrDrff = 9,
    TanhGrad = 10,
    ReluGrad = 11,
    RsqrtGrad = 12,
}

pub const OP_MUL: u32 = 0;
pub const OP_ADD: u32 = 1;
pub const OP_SUB: u32 = 2;
pub const OP_DIV: u32 = 3;
pub const OP_DIV_NO_NAN: u32 = 4;
pub const OP_DIV_REAL: u32 = 5;
pub const OP_MAX: u32 = 6;
pub const OP_MIN: u32 = 7;
pub const OP_POW: u32 = 8;
pub const OP_SQR_DIFF: u32 = 9;
pub const OP_TANH_GRAD: u32 = 10;
pub const OP_RELU_GRAD: u32 = 11;
pub const OP_RSQRT_GRAD: u32 = 12;

impl TryFrom<u32> for BinaryOp {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            OP_MUL => Ok(Self::Mul),
            OP_ADD => Ok(Self::Add),
            OP_SUB => Ok(Self::Sub),
            OP_DIV => Ok(Self::Div),
            OP_DIV_NO_NAN => Ok(Self::DivNoNan),
            OP_DIV_REAL => Ok(Self::DivReal),
            OP_MAX => Ok(Self::Max),
            OP_MIN => Ok(Self::Min),
            OP_POW => Ok(Self::Pow),
            OP_SQR_DIFF => Ok(Self::SqrDrff),
            OP_TANH_GRAD => Ok(Self::TanhGrad),
            OP_RELU_GRAD => Ok(Self::ReluGrad),
            OP_RSQRT_GRAD => Ok(Self::RsqrtGrad),
            _ => Err(()),
        }
    }
}

pub enum Version {
    Broad,
    NoBroad,
    Simple,
}

pub trait BinaryKernelVersion<'a> {
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

pub struct BinaryKernel<'a> {
    inst: &'a VultenInstance,
    d_type: VultenDataType,
    op: BinaryOp,
    a: Option<KernelBuff<'a>>,
    a_dims: Vec<i64>,
    b: Option<KernelBuff<'a>>,
    b_dims: Vec<i64>,
    output: Option<KernelBuff<'a>>,
    output_dims: Vec<i64>,
    needs_boardcast: bool,
    simple_boardcast: bool,
}

impl<'a> BinaryKernel<'a> {
    pub fn new(
        inst: &'a VultenInstance,
        d_type: VultenDataType,
        op: BinaryOp,
        shape_helper: BroadcastShapeHelper,
    ) -> Self {
        Self {
            inst,
            d_type,
            op,
            a: Default::default(),
            a_dims: shape_helper.a_padded,
            b: Default::default(),
            b_dims: shape_helper.b_padded,
            output: Default::default(),
            output_dims: shape_helper.out_shape,
            needs_boardcast: shape_helper.needs_boardcast,
            simple_boardcast: shape_helper.simple_boardcast,
        }
    }

    pub fn a(mut self, buff: KernelBuff<'a>) -> Result<Self, &'static str> {
        self.a = Some(buff);

        Ok(self)
    }

    pub fn b(mut self, buff: KernelBuff<'a>) -> Result<Self, &'static str> {
        self.b = Some(buff);

        Ok(self)
    }

    pub fn output(mut self, buff: KernelBuff<'a>) -> Result<Self, &'static str> {
        self.output = Some(buff);

        Ok(self)
    }

    pub fn build(
        self,
        ver_override: Option<Version>,
    ) -> Result<Box<dyn BinaryKernelVersion<'a> + 'a>, &'static str> {
        match ver_override {
            Some(ver_override) => match ver_override {
                Version::Broad => Ok(Box::new(BinaryKernelBroad::new(self))),
                Version::NoBroad => Ok(Box::new(BinaryKernelNoBroad::new(self))),
                Version::Simple => Ok(Box::new(BinaryKernelSimple::new(self))),
            },
            None => {
                if !self.needs_boardcast {
                    Ok(Box::new(BinaryKernelNoBroad::new(self)))
                } else if self.simple_boardcast {
                    Ok(Box::new(BinaryKernelSimple::new(self)))
                } else {
                    Ok(Box::new(BinaryKernelBroad::new(self)))
                }
            }
        }
    }
}
