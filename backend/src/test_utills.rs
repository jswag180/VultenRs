use crate::{VultenDataType, VultenInstance};

pub fn get_test_instance() -> VultenInstance {
    let dev_num: Option<usize> = match std::env::var("VULTEN_TEST_DEV") {
        Ok(val) => Some(val.parse().unwrap_or_default()),
        Err(_) => None,
    };

    VultenInstance::new(dev_num)
}

pub fn buff_size_from_dims(dims: &[i64], dtype: VultenDataType) -> u64 {
    dims.iter().product::<i64>() as u64 * dtype.size_of().unwrap() as u64
}
