pub mod binary_broad;
pub mod binary_no_board;
pub mod binary_simple;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
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
