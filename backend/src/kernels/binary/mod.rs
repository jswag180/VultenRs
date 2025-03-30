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
    Mul,
    Add,
    Sub,
    Div,
    DivNoNan,
    DivReal,
    Max,
    Min,
    Pow,
    SqrDrff,
    TanhGrad,
    ReluGrad,
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
            _ => Err(()),
        }
    }
}

impl From<BinaryOp> for u32 {
    fn from(value: BinaryOp) -> Self {
        match value {
            BinaryOp::Mul => OP_MUL,
            BinaryOp::Add => OP_ADD,
            BinaryOp::Sub => OP_SUB,
            BinaryOp::Div => OP_DIV,
            BinaryOp::DivNoNan => OP_DIV_NO_NAN,
            BinaryOp::DivReal => OP_DIV_REAL,
            BinaryOp::Max => OP_MAX,
            BinaryOp::Min => OP_MIN,
            BinaryOp::Pow => OP_POW,
            BinaryOp::SqrDrff => OP_SQR_DIFF,
            BinaryOp::TanhGrad => OP_TANH_GRAD,
            BinaryOp::ReluGrad => OP_RELU_GRAD,
        }
    }
}

impl BinaryOp {
    pub const fn into_u32(self) -> u32 {
        match self {
            Self::Mul => OP_MUL,
            Self::Add => OP_ADD,
            Self::Sub => OP_SUB,
            Self::Div => OP_DIV,
            Self::DivNoNan => OP_DIV_NO_NAN,
            Self::DivReal => OP_DIV_REAL,
            Self::Max => OP_MAX,
            Self::Min => OP_MIN,
            Self::Pow => OP_POW,
            Self::SqrDrff => OP_SQR_DIFF,
            Self::TanhGrad => OP_TANH_GRAD,
            Self::ReluGrad => OP_RELU_GRAD,
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
