pub mod conv2d;

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
        return Err(format!("Stride must be > 0 got {:?}", stride));
    }
    if dilation_rate <= 0 {
        return Err(format!("Dilation must be > 0 got {:?}", dilation_rate));
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
            return Err(format!("Padding format not supported {:?}", padding));
        }
    }

    if *output_size < 0 {
        return Err(format!("Resulting size would be negative output: {:?} input: {:?} effective_filter_size: {:?} stride: {:?}", *output_size, input_size, effective_filter_size, stride));
    }

    Ok(())
}
