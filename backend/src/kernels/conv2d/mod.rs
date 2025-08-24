use std::sync::Arc;

use conv2d_gemm::Conv2DKernelGemm;

use crate::{
    cmd_buff::CommandBufferBuilder, descriptor::VultenDescriptor, pipeline::VultenPipeline,
    VultenDataType, VultenInstance,
};

use super::{ChannelFormat, KernelBuff};

pub mod backprop_filter;
pub mod backprop_input;
pub mod col2im;
pub mod conv2d_gemm;
pub mod im2col;

#[derive(Debug, Default, PartialEq, Eq)]
pub enum Padding {
    #[default]
    Valid,
    Same,
    Explicit,
}

impl TryFrom<&str> for Padding {
    type Error = &'static str;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "VALID" => Ok(Self::Valid),
            "SAME" => Ok(Self::Same),
            "EXPLICIT" => Ok(Self::Explicit),
            _ => Err("Invalid format!"),
        }
    }
}

pub fn get_windowed_ouput(
    input_size: i64,
    filter_size: i64,
    dilation_rate: i64,
    stride: i64,
    padding: &Padding,
    output_size: &mut i64,
    padding_before: &mut i64,
) -> Result<(), String> {
    if stride <= 0 {
        return Err(format!("Stride must be > 0 got {stride:?}"));
    }
    if dilation_rate <= 0 {
        return Err(format!("Dilation must be > 0 got {dilation_rate:?}"));
    }

    let effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    match padding {
        Padding::Valid => {
            *output_size = (input_size - effective_filter_size + stride) / stride;
            *padding_before = 0;
        }
        Padding::Same => {
            *output_size = (input_size + stride - 1) / stride;
            let padding_needed = i64::max(
                0,
                (*output_size - 1) * stride + effective_filter_size - input_size,
            );
            // For odd values of total padding, add more padding at the 'right'
            // side of the given dimension.
            *padding_before = padding_needed / 2;
        }
        _ => {
            return Err(format!("Padding format not supported {padding:?}"));
        }
    }

    if *output_size < 0 {
        return Err(format!("Resulting size would be negative output: {:?} input: {:?} effective_filter_size: {:?} stride: {:?}", *output_size, input_size, effective_filter_size, stride));
    }

    Ok(())
}

pub enum Version {
    Gemm,
}

pub trait Conv2DKernelVersion<'a> {
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

pub struct Conv2DKernel<'a> {
    inst: &'a VultenInstance,
    d_type: VultenDataType,
    padd_h: u32,
    padd_w: u32,
    stride_x: u32,
    stride_y: u32,
    dilation_x: u32,
    dilation_y: u32,
    format: ChannelFormat,
    filters: Option<KernelBuff<'a>>,
    filter_dims: Option<&'a [i32]>,
    input: Option<KernelBuff<'a>>,
    input_dims: Option<&'a [i64]>,
    output: Option<KernelBuff<'a>>,
    output_dims: Option<&'a [i64]>,
}

impl<'a> Conv2DKernel<'a> {
    pub fn new(
        inst: &'a VultenInstance,
        d_type: VultenDataType,
        padding: (u32, u32),
        strides: (u32, u32),
        dilations: (u32, u32),
        format: ChannelFormat,
    ) -> Self {
        Self {
            inst,
            d_type,
            padd_h: padding.0,
            padd_w: padding.1,
            stride_x: strides.0,
            stride_y: strides.1,
            dilation_x: dilations.0,
            dilation_y: dilations.1,
            format,
            filters: Default::default(),
            filter_dims: Default::default(),
            input: Default::default(),
            input_dims: Default::default(),
            output: Default::default(),
            output_dims: Default::default(),
        }
    }

    pub fn filter(
        mut self,
        filters: KernelBuff<'a>,
        dims: &'a [i32],
    ) -> Result<Self, &'static str> {
        self.filter_dims = Some(dims);
        self.filters = Some(filters);

        Ok(self)
    }

    pub fn input(mut self, input: KernelBuff<'a>, dims: &'a [i64]) -> Result<Self, &'static str> {
        self.input_dims = Some(dims);
        self.input = Some(input);

        Ok(self)
    }

    pub fn output(mut self, output: KernelBuff<'a>, dims: &'a [i64]) -> Result<Self, &'static str> {
        self.output_dims = Some(dims);
        self.output = Some(output);

        Ok(self)
    }

    pub fn build(
        self,
        ver_override: Option<Version>,
    ) -> Result<Box<dyn Conv2DKernelVersion<'a> + 'a>, &'static str> {
        match ver_override {
            Some(ver_override) => match ver_override {
                Version::Gemm => Ok(Box::new(Conv2DKernelGemm::new(self)?)),
            },
            None => Ok(Box::new(Conv2DKernelGemm::new(self)?)),
        }
    }
}
