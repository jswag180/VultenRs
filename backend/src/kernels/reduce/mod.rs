use std::num::TryFromIntError;

pub mod reduce;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
}

const OP_SUM: u32 = 0;
const OP_MAX: u32 = 1;
const OP_MIN: u32 = 2;
const OP_MEAN: u32 = 3;

impl TryFrom<u32> for ReduceOp {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            OP_SUM => Ok(Self::Sum),
            OP_MAX => Ok(Self::Max),
            OP_MIN => Ok(Self::Min),
            OP_MEAN => Ok(Self::Mean),
            _ => Err(()),
        }
    }
}

impl From<ReduceOp> for u32 {
    fn from(value: ReduceOp) -> Self {
        match value {
            ReduceOp::Sum => OP_SUM,
            ReduceOp::Max => OP_MAX,
            ReduceOp::Min => OP_MIN,
            ReduceOp::Mean => OP_MEAN,
        }
    }
}

impl ReduceOp {
    pub const fn into_u32(self) -> u32 {
        match self {
            Self::Sum => OP_SUM,
            Self::Max => OP_MAX,
            Self::Min => OP_MIN,
            Self::Mean => OP_MEAN,
        }
    }
}

pub fn process_dims<T: Copy + std::fmt::Debug + Into<i64>>(
    dims: &Vec<i64>,
    reduce_dims: &[T],
) -> Result<Vec<u32>, TryFromIntError>
where
    u32: TryFrom<T, Error = TryFromIntError>,
{
    let mut new_dims: Vec<u32> = Vec::new();

    //Process wrap around dims eg. Arr: [1, 2, 3] Dims: [-1] -> Dims: [2]
    //and convert to u32
    for dim in reduce_dims {
        if <T as Into<i64>>::into(*dim) < 0 {
            new_dims.push((<T as TryInto<i64>>::try_into(*dim)? + dims.len() as i64).try_into()?);
        } else {
            new_dims.push((*dim).try_into()?);
        }
    }

    new_dims.sort();

    Ok(new_dims)
}
