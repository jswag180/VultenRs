#[inline]
pub fn calculate_strdies(dims: &[i64]) -> Vec<i64> {
    let mut strides: Vec<i64> = Vec::new();
    strides.resize_with(dims.len() + 1, || 1);
    for (i, srid) in strides.iter_mut().enumerate().take(dims.len()) {
        for j in dims.iter().skip(i) {
            *srid *= j;
        }
    }

    strides.to_vec()
}
