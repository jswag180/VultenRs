use std::{num::TryFromIntError, sync::Arc};

use reduce_slow::ReduceKernelSlow;

use crate::{
    cmd_buff::CommandBufferBuilder, descriptor::VultenDescriptor, pipeline::VultenPipeline,
    VultenDataType, VultenInstance,
};

use super::KernelBuff;

pub mod reduce_slow;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ReduceOp {
    Sum = 0,
    Max = 1,
    Min = 2,
    Mean = 3,
    Prod = 4,
}

const OP_SUM: u32 = 0;
const OP_MAX: u32 = 1;
const OP_MIN: u32 = 2;
const OP_MEAN: u32 = 3;
const OP_PROD: u32 = 4;

impl TryFrom<u32> for ReduceOp {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            OP_SUM => Ok(Self::Sum),
            OP_MAX => Ok(Self::Max),
            OP_MIN => Ok(Self::Min),
            OP_MEAN => Ok(Self::Mean),
            OP_PROD => Ok(Self::Prod),
            _ => Err(()),
        }
    }
}

pub fn process_dims<T: Copy + std::fmt::Debug + Into<i64>>(
    dims: &[i64],
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
            new_dims.push((<T as Into<i64>>::into(*dim) + dims.len() as i64).try_into()?);
        } else {
            new_dims.push((*dim).try_into()?);
        }
    }

    new_dims.sort();

    Ok(new_dims)
}

pub enum Version {
    Slow,
}

pub trait ReduceKernelVersion<'a> {
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

pub struct ReduceKernel<'a> {
    inst: &'a VultenInstance,
    d_type: VultenDataType,
    op: ReduceOp,
    reduce_dims: Vec<u32>,
    input: Option<KernelBuff<'a>>,
    input_dims: Option<&'a [i64]>,
    output: Option<KernelBuff<'a>>,
    output_dims: Option<&'a [i64]>,
}

impl<'a> ReduceKernel<'a> {
    pub fn new(inst: &'a VultenInstance, d_type: VultenDataType, op: ReduceOp) -> Self {
        Self {
            inst,
            d_type,
            op,
            reduce_dims: Default::default(),
            input: Default::default(),
            input_dims: Default::default(),
            output: Default::default(),
            output_dims: Default::default(),
        }
    }

    pub fn reduce_dims(mut self, dims: Vec<u32>) -> Result<Self, &'static str> {
        self.reduce_dims = dims;

        Ok(self)
    }

    pub fn input(mut self, buff: KernelBuff<'a>, dims: &'a [i64]) -> Result<Self, &'static str> {
        if dims.contains(&0) {
            return Err("Input has a zero dim!");
        }
        self.input = Some(buff);
        self.input_dims = Some(dims);

        Ok(self)
    }

    pub fn output(mut self, buff: KernelBuff<'a>, dims: &'a [i64]) -> Result<Self, &'static str> {
        self.output = Some(buff);
        self.output_dims = Some(dims);

        Ok(self)
    }

    pub fn build(
        self,
        ver_override: Option<Version>,
    ) -> Result<Box<dyn ReduceKernelVersion<'a> + 'a>, &'static str> {
        match ver_override {
            Some(ver_override) => match ver_override {
                Version::Slow => Ok(Box::new(ReduceKernelSlow::new(self)?)),
            },
            None => Ok(Box::new(ReduceKernelSlow::new(self)?)),
        }
    }
}
