use std::ops::Index;

#[derive(Debug)]
pub enum Dims<'a, T> {
    Vec(Vec<T>),
    Slice(&'a [T]),
}

impl<T> Index<usize> for Dims<'_, T> {
    type Output = T;

    fn index(&self, idx: usize) -> &<Self as Index<usize>>::Output {
        match self {
            Dims::Vec(arr) => &arr[idx],
            Dims::Slice(arr) => &arr[idx],
        }
    }
}
