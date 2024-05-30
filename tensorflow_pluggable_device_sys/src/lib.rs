#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub type TF_Bool = u8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let sat = unsafe { TF_NewStatus() };
    }
}
