use std::sync::{Arc, Mutex};

use arc_swap::Guard;
use ash::{
    prelude::VkResult,
    vk::{self, DescriptorPool, DescriptorPoolCreateFlags, DescriptorSet},
};

use crate::{pipeline::VultenPipeline, VultenInstance};

//This asumes that nothing will every ask for more sets then it.
const POOL_SIZE: u32 = 16;

type PoolGuard = Guard<Arc<Vec<Arc<Mutex<DescriptorPool>>>>>;

pub struct VultenDescriptor<'a> {
    inst: &'a VultenInstance,
    pub descriptor: Vec<DescriptorSet>,
    pool: Arc<Mutex<DescriptorPool>>,
}

impl Drop for VultenDescriptor<'_> {
    fn drop(&mut self) {
        unsafe {
            self.inst
                .device
                .free_descriptor_sets(*self.pool.lock().unwrap(), self.descriptor.as_slice())
                .unwrap();
        }
    }
}

impl VultenInstance {
    pub fn get_descriptor_set(
        &self,
        buff_type: vk::DescriptorType,
        pipeline: Arc<VultenPipeline>,
    ) -> VkResult<VultenDescriptor> {
        let pools = self.descriptor_pools.load();

        let layout = [pipeline.descriptor_set_layout];

        for pool in pools.iter() {
            let locked_pool = pool.lock().unwrap(); // This should prob be try_lock and then continue if not locked

            let disc_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(*locked_pool)
                .set_layouts(&layout);

            match unsafe { self.device.allocate_descriptor_sets(&disc_alloc_info) } {
                Ok(i) => {
                    return Ok(VultenDescriptor {
                        inst: self,
                        descriptor: i,
                        pool: pool.clone(),
                    });
                }
                Err(_) => {
                    continue;
                }
            };
        }

        //There are no pools so create new pool
        let descriptor_pool = self.allocate_new_pool(buff_type)?;

        //if we can't allocate a set from a fresh pool something is very wrong
        let disc_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layout);
        let new_descriptor = unsafe { self.device.allocate_descriptor_sets(&disc_alloc_info)? };

        //sense the pool it good to go add it to the pools
        let new_pool = self.add_pool_to_pools(descriptor_pool, Some(pools));

        Ok(VultenDescriptor {
            inst: self,
            descriptor: new_descriptor,
            pool: new_pool,
        })
    }

    fn allocate_new_pool(&self, buff_type: vk::DescriptorType) -> VkResult<DescriptorPool> {
        let pool_size = [vk::DescriptorPoolSize::default()
            .descriptor_count(POOL_SIZE)
            .ty(buff_type)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(POOL_SIZE)
            .pool_sizes(&pool_size)
            .flags(DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        unsafe { self.device.create_descriptor_pool(&pool_info, None) }
    }

    fn add_pool_to_pools(
        &self,
        pool: DescriptorPool,
        pools_guard: Option<PoolGuard>,
    ) -> Arc<Mutex<DescriptorPool>> {
        let pools = match pools_guard {
            Some(i) => i,
            None => self.descriptor_pools.load(),
        };

        let mut pools_copy = Vec::clone(&pools);
        let new_pool = Arc::new(Mutex::new(pool));
        pools_copy.push(new_pool.clone());
        self.descriptor_pools.store(pools_copy.into());

        new_pool
    }
}
