use std::ops::Index;

#[derive(Debug)]
pub enum Dims<'a, T: Clone> {
    Vec(Vec<T>),
    Slice(&'a [T]),
}

impl<T: Clone> Index<usize> for Dims<'_, T> {
    type Output = T;

    fn index(&self, idx: usize) -> &<Self as Index<usize>>::Output {
        match self {
            Dims::Vec(arr) => &arr[idx],
            Dims::Slice(arr) => &arr[idx],
        }
    }
}

impl<T: Clone> Dims<'_, T> {
    pub fn to_vec(&self) -> Vec<T> {
        match self {
            Dims::Vec(arr) => arr.to_vec(),
            Dims::Slice(arr) => arr.to_vec(),
        }
    }
}
