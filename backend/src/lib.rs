use arc_swap::ArcSwap;
use ash::{
    prelude::VkResult,
    vk::{
        self, CommandPoolCreateInfo, CopyDescriptorSet, DescriptorPool, Fence, FenceCreateInfo,
        PhysicalDeviceProperties2, PhysicalDeviceSubgroupProperties, PipelineCacheCreateInfo,
        WriteDescriptorSet,
    },
    Entry,
};
pub use ash::{Device, Instance};
use core::mem::ManuallyDrop;
use memory::VultenBuffer;
use pipeline::{PipelineSpecs, VultenPipeline};
use std::{
    collections::HashMap,
    ffi::{c_char, c_void, CStr},
    hash::Hash,
    sync::{Arc, Mutex, RwLock},
};
use va::Va;

const VK_API_VER: u32 = vk::make_api_version(0, 1, 3, 0);
const VK_ENV_VER: u32 = shaderc::EnvVersion::Vulkan1_3 as u32;

pub mod cmd_buff;
pub mod compiler;
pub mod descriptor;
pub mod memory;
pub mod pipeline;
pub mod queue;
pub mod utills;
pub mod va;

pub mod kernels;

pub static mut GOLBAL_DEVICE_VA: Va<Arc<VultenBuffer>> = Va::new();
pub static mut GLOBAL_INSTANCES: RwLock<Vec<*mut VultenInstance>> = RwLock::new(Vec::new());

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct VultenDataType(u32);
pub const DT_FLOAT: VultenDataType = VultenDataType(1);
pub const DT_INT32: VultenDataType = VultenDataType(3);
pub const DT_UINT32: VultenDataType = VultenDataType(22);
pub const DT_INT64: VultenDataType = VultenDataType(9);
pub const DT_UINT64: VultenDataType = VultenDataType(23);

impl From<u32> for VultenDataType {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

pub struct DeviceProperties {
    pub max_work_group: [u32; 3],
    pub max_work_group_invo: u32,
    pub max_work_group_invo_size: [u32; 3],
    pub sub_group_size: u32,
}

pub struct VultenInstance {
    #[allow(dead_code)]
    entry: Entry, // This cannot be droped
    pub dev_num: u64,
    vk_instance: Instance,
    physical_device: vk::PhysicalDevice,
    pub device: Device,
    queues: Vec<Arc<Mutex<queue::VultenQueue>>>,
    allocator: ManuallyDrop<vk_mem::Allocator>,
    pipeline_cache: ManuallyDrop<vk::PipelineCache>,
    extens: Vec<*const c_char>,
    pipelines: parking_lot::RwLock<HashMap<PipelineSpecs, Arc<VultenPipeline>>>,
    descriptor_pools: ArcSwap<Vec<Arc<Mutex<DescriptorPool>>>>,
    pub device_props: DeviceProperties,
}

unsafe impl Send for VultenInstance {}

impl Drop for VultenInstance {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_cache(*self.pipeline_cache, None);
            ManuallyDrop::drop(&mut self.pipeline_cache);
            for q in self.queues.iter() {
                let queue = q.lock().unwrap();
                self.device.destroy_command_pool(queue.pool, None);
            }
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.vk_instance.destroy_instance(None)
        };
    }
}

impl VultenInstance {
    pub fn new(device_num: Option<usize>) -> Self {
        let entry = Entry::linked(); //unsafe { Entry::load() }.expect("Error loading vulkan loader lib!");
        let inst = create_instance(&entry);

        let physical_devices =
            unsafe { inst.enumerate_physical_devices() }.expect("Error no device");
        let dev_num = device_num.unwrap_or(0);
        let physical_device = physical_devices[dev_num];

        let queue_propertys =
            unsafe { inst.get_physical_device_queue_family_properties(physical_device) };

        let mut queue_flags: Vec<(vk::QueueFlags, u32)> = Vec::new();
        let mut queue_infos: Vec<vk::DeviceQueueCreateInfo> = Vec::new();
        let priorities: Vec<f32> = vec![
            1.0;
            queue_propertys
                .clone()
                .into_iter()
                .map(|x| { x.queue_count })
                .reduce(|acc, e| { u32::max(acc, e) })
                .unwrap_or_default() as usize
        ];
        for (i, q) in queue_propertys.clone().into_iter().enumerate() {
            let queue_info = vk::DeviceQueueCreateInfo {
                queue_family_index: i as u32,
                queue_count: q.queue_count,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            };
            queue_infos.push(queue_info);

            for _ in 0..q.queue_count {
                queue_flags.push((q.queue_flags, i as u32));
            }
        }

        let availble_extens = unsafe {
            inst.enumerate_device_extension_properties(physical_device)
                .unwrap()
        };
        let mut extens: Vec<*const c_char> = Vec::new();

        let have_memory_budget = enable_if_availble(
            c"VK_EXT_memory_budget".as_ptr(),
            &mut extens,
            &availble_extens,
        );

        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::builder().maintenance4(true);
        let feat = vk::PhysicalDeviceFeatures::builder()
            .shader_int64(true)
            .build();
        let feat2 = vk::PhysicalDeviceFeatures2::builder()
            .features(feat)
            .push_next(&mut maintenance4)
            .build();

        let device_create_info = vk::DeviceCreateInfo {
            p_queue_create_infos: queue_infos.as_ptr(),
            queue_create_info_count: queue_infos.len() as u32,
            pp_enabled_extension_names: extens.as_ptr(),
            enabled_extension_count: extens.len() as u32,
            p_next: &feat2 as *const vk::PhysicalDeviceFeatures2 as *const c_void,
            ..Default::default()
        };

        let device = unsafe { inst.create_device(physical_device, &device_create_info, None) }
            .expect("Error could not create device!");

        let mut queues: Vec<Arc<Mutex<queue::VultenQueue>>> = Vec::new();
        for (i, qp) in queue_propertys.into_iter().enumerate() {
            for qc in 0..qp.queue_count {
                let q = unsafe { device.get_device_queue(i as u32, qc) };
                let q_pool_info = CommandPoolCreateInfo::builder().queue_family_index(i as u32);
                let pool = unsafe { device.create_command_pool(&q_pool_info, None) }
                    .expect("Error filed to create command pool for queue!");
                queues.push(Arc::new(Mutex::new(queue::VultenQueue::new(
                    q,
                    qp.queue_flags,
                    pool,
                ))));
            }
        }

        let mut allocator_flags = vk_mem::AllocatorCreateFlags::empty();
        if have_memory_budget {
            allocator_flags |= vk_mem::AllocatorCreateFlags::EXT_MEMORY_BUDGET;
        }

        let allocator_create_info =
            vk_mem::AllocatorCreateInfo::new(&inst, &device, physical_device)
                .vulkan_api_version(VK_API_VER)
                .flags(allocator_flags);
        let allocator = vk_mem::Allocator::new(allocator_create_info)
            .expect("Error could no create allocator!");

        let pipeline_cache_info = PipelineCacheCreateInfo::builder().build();
        let pipeline_cache =
            unsafe { device.create_pipeline_cache(&pipeline_cache_info, None) }.unwrap();

        let mut sub_props = PhysicalDeviceSubgroupProperties::default();
        let mut props = PhysicalDeviceProperties2::builder()
            .push_next(&mut sub_props)
            .build();
        unsafe { inst.get_physical_device_properties2(physical_device, &mut props) };

        let device_props = DeviceProperties {
            max_work_group: props.properties.limits.max_compute_work_group_count,
            max_work_group_invo: props.properties.limits.max_compute_work_group_invocations,
            max_work_group_invo_size: props.properties.limits.max_compute_work_group_size,
            sub_group_size: sub_props.subgroup_size,
        };

        VultenInstance {
            entry,
            dev_num: dev_num as u64,
            vk_instance: inst,
            physical_device,
            device,
            queues,
            allocator: ManuallyDrop::new(allocator),
            pipeline_cache: ManuallyDrop::new(pipeline_cache),
            extens,
            pipelines: HashMap::new().into(),
            descriptor_pools: ArcSwap::from_pointee(Vec::new()),
            //descriptor_sets: ArcSwap::from_pointee(Vec::new()),
            device_props,
        }
    }

    pub fn get_num_devices() -> i32 {
        let entry = Entry::linked(); //unsafe { Entry::load() }.expect("Error loading vulkan loader lib!");
        let appinfo: vk::ApplicationInfo = vk::ApplicationInfo {
            application_version: 0,
            api_version: VK_API_VER,
            ..Default::default()
        };

        let instance_info = vk::InstanceCreateInfo::builder()
            .application_info(&appinfo)
            .build();

        let inst = unsafe { entry.create_instance(&instance_info, None) }
            .expect("Error failed to create vkInstance!");

        unsafe { inst.enumerate_physical_devices() }
            .expect("Error no device")
            .len() as i32
    }

    pub fn get_device_name(&self) -> *const c_char {
        unsafe {
            self.vk_instance
                .get_physical_device_properties(self.physical_device)
                .device_name
                .as_ptr()
        }
    }

    pub fn update_descriptor_sets(
        &self,
        descriptors: &[WriteDescriptorSet],
        descriptor_cpys: &[CopyDescriptorSet],
    ) {
        unsafe {
            self.device
                .update_descriptor_sets(descriptors, descriptor_cpys);
        }
    }

    pub fn create_fence(&self) -> VkResult<Fence> {
        unsafe { self.device.create_fence(&FenceCreateInfo::default(), None) }
    }

    pub fn wait_for_fences(&self, fences: &[Fence], wait_all: bool) -> VkResult<()> {
        unsafe { self.device.wait_for_fences(fences, wait_all, u64::MAX) }
    }

    pub fn destroy_fence(&self, fence: Fence) {
        unsafe {
            self.device.destroy_fence(fence, None);
        }
    }
}

fn create_instance(entry: &Entry) -> Instance {
    let appinfo: vk::ApplicationInfo = vk::ApplicationInfo {
        application_version: 0,
        api_version: VK_API_VER,
        ..Default::default()
    };

    let mut layers: Vec<*const c_char> = Vec::new();

    match std::env::var("VULTEN_VALIDATION")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "true" | "on" => layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr()),
        _ => (),
    }

    let instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&appinfo)
        .enabled_layer_names(&layers)
        .build();

    unsafe { entry.create_instance(&instance_info, None) }
        .expect("Error failed to create vkInstance!")
}

fn enable_if_availble(
    exten: *const c_char,
    extens: &mut Vec<*const c_char>,
    availble_extens: &[vk::ExtensionProperties],
) -> bool {
    let availble = availble_extens.iter().find(|&&x| {
        let wanted_exten_str =
            unsafe { std::str::from_utf8_unchecked(CStr::from_ptr(exten as *const _).to_bytes()) };
        let exten_str = unsafe {
            std::str::from_utf8_unchecked(
                CStr::from_ptr(x.extension_name.as_ptr() as *const _).to_bytes(),
            )
        };

        wanted_exten_str == exten_str
    });

    if availble.is_some() {
        extens.push(exten);

        true
    } else {
        false
    }
}
