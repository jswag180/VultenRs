use std::{
    fmt::{Debug, Pointer},
    os::raw::c_void,
    sync::{atomic::AtomicU64, RwLock},
};

/// The top 4 bits are used for device id and the lower 60 for addressing
#[derive(Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct VaAddress(u64);

const ADDRESS_MASK: VaAddress =
    VaAddress(0b0000111111111111111111111111111111111111111111111111111111111111);

impl Pointer for VaAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:p}", self.0 as *const u64))
    }
}

impl Debug for VaAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:p}", self.0 as *const u64))
    }
}

impl From<u64> for VaAddress {
    fn from(value: u64) -> Self {
        VaAddress(value)
    }
}

impl From<*mut c_void> for VaAddress {
    fn from(value: *mut c_void) -> Self {
        VaAddress(value as u64)
    }
}

impl PartialEq<u64> for VaAddress {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl std::ops::Add for VaAddress {
    type Output = Self;
    fn add(self, rhs: VaAddress) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}

impl std::ops::Sub for VaAddress {
    type Output = Self;
    fn sub(self, rhs: VaAddress) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl std::ops::BitAnd for VaAddress {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        (self.0 & rhs.0).into()
    }
}

impl VaAddress {
    pub const fn new(addr: u64) -> Self {
        Self(addr)
    }

    pub fn raw_ptr(&self) -> *mut c_void {
        self.0 as *mut c_void
    }

    #[inline]
    pub fn get_device_num(addr: VaAddress) -> u64 {
        addr.0 >> 60
    }

    #[inline]
    fn remove_device_bits(addr: VaAddress) -> VaAddress {
        addr & ADDRESS_MASK
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct VaAlloc<T: Clone> {
    pub obj: T,
    pub addr: VaAddress,
    pub size: u64,
}

pub struct Va<T: Clone> {
    //dev_num: VaAddres,
    allocs: RwLock<Vec<VaAlloc<T>>>,
    last_alloc_addr: AtomicU64,
}

impl<T: Clone> Default for Va<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Va<T> {
    pub const fn new() -> Self {
        Self {
            //dev_num: dev_num.into(),
            allocs: RwLock::new(Vec::new()),
            last_alloc_addr: AtomicU64::new(0),
        }
    }

    pub fn find_va(&self, addr: VaAddress) -> Result<(VaAlloc<T>, u64), &'static str> {
        for alloc in self.allocs.read().unwrap().iter() {
            if addr >= alloc.addr && addr <= (alloc.addr + (alloc.size - 1).into()) {
                return Ok((alloc.clone(), (addr - alloc.addr).0));
            }
        }

        Err("No matching alloc")
    }

    #[inline]
    fn align_to(start: VaAddress, alignment: u64) -> VaAddress {
        (((start.0) + ((alignment) - 1)) & !((alignment) - 1)).into()
    }

    pub fn alloc(&self, dev_num: u64, obj: T, size: u64) -> Result<VaAddress, &'static str> {
        if size == 0 {
            return Err("Size must be > 0");
        }
        let mut allocs = self.allocs.write().unwrap();
        let mut new_addr: VaAddress = (dev_num << 60).into();
        let last_alloc_addr: VaAddress = self
            .last_alloc_addr
            .load(std::sync::atomic::Ordering::Relaxed)
            .into();

        //If this is the first alloc just start at 64
        if last_alloc_addr == 0 {
            new_addr = new_addr + 64.into();
            allocs.push(VaAlloc {
                obj,
                addr: new_addr,
                size,
            });
            self.last_alloc_addr
                .store((size + 64).into(), std::sync::atomic::Ordering::Relaxed);
            debug_assert!((new_addr.0 as *mut u64).align_offset(64) == 0);
            return Ok(new_addr);
        }

        //look for space in-between allocs
        for i in 0..(allocs.len().max(1) - 1) {
            let aligned_start = Self::align_to(
                VaAddress::remove_device_bits(allocs[i].addr) + allocs[i].size.into(),
                64,
            );
            if (VaAddress::remove_device_bits(allocs[i + 1].addr) - aligned_start) >= size.into() {
                new_addr = new_addr + aligned_start;
                allocs.insert(
                    i + 1,
                    VaAlloc {
                        obj,
                        addr: new_addr,
                        size,
                    },
                );
                debug_assert!((new_addr.0 as *mut u64).align_offset(64) == 0);
                return Ok(new_addr);
            }
        }

        //use the end of the last alloc for the new one
        //if the new alloc will overflow into the device id bits err out
        //ADDRESS_MASK is is also the highest address
        if last_alloc_addr + size.into() > ADDRESS_MASK {
            return Err("Out of address space");
        }

        let alignment_offset = Self::align_to(last_alloc_addr, 64);
        new_addr = new_addr + alignment_offset;
        allocs.push(VaAlloc {
            obj,
            addr: new_addr,
            size,
        });
        self.last_alloc_addr.store(
            alignment_offset.0 + size,
            std::sync::atomic::Ordering::Relaxed,
        );
        debug_assert!((new_addr.0 as *mut u64).align_offset(64) == 0);
        Ok(new_addr)
    }

    pub fn free(&self, addr: VaAddress) -> Result<(), &'static str> {
        let mut allocs = self.allocs.write().unwrap();
        for i in 0..allocs.len() {
            if addr == allocs[i].addr {
                allocs.remove(i);
                return Ok(());
            }
        }
        Err("Failed to find va to free")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //All allocations should be 64 bit aligned
    #[test]
    fn alignment() {
        let mut allocator: Va<i32> = Va::new();

        let alloc_1 = allocator.alloc(0, 1, 1).unwrap();
        assert_eq!((alloc_1.0 as *mut u64).align_offset(64), 0);

        let alloc_2 = allocator.alloc(0, 2, 1).unwrap();
        assert_eq!((alloc_2.0 as *mut u64).align_offset(64), 0);

        let alloc_3 = allocator.alloc(0, 3, 128).unwrap();
        assert_eq!((alloc_3.0 as *mut u64).align_offset(64), 0);

        let alloc_4 = allocator.alloc(0, 4, 1).unwrap();
        assert_eq!((alloc_4.0 as *mut u64).align_offset(64), 0);

        allocator.free(alloc_3).unwrap();

        let alloc_5 = allocator.alloc(0, 5, 1).unwrap();
        assert_eq!((alloc_5.0 as *mut u64).align_offset(64), 0);
    }
}
