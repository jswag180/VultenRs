use arc_swap::ArcSwap;
use ash::{
    prelude::VkResult,
    vk::{
        self, CommandPoolCreateInfo, CopyDescriptorSet, DescriptorPool, Fence, FenceCreateInfo,
        PhysicalDevice, PhysicalDeviceProperties2, PhysicalDeviceSubgroupProperties,
        PipelineCacheCreateInfo, WriteDescriptorSet,
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
    sync::{Arc, LazyLock, Mutex, RwLock},
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

pub static GOLBAL_DEVICE_VA: Va<Arc<VultenBuffer>> = Va::new();
pub static GLOBAL_INSTANCES: RwLock<Vec<Arc<VultenInstance>>> = RwLock::new(Vec::new());
pub static ENV_SETTINGS: LazyLock<EnvSettings> = LazyLock::new(|| {
    let env_var = std::env::var("VULTEN_SETTINGS");
    match env_var {
        Ok(vars) => {
            let mut settings = EnvSettings::default();

            if vars.contains("DISABLE_INT64") {
                settings.disable_int64 = true;
            }
            if vars.contains("DISABLE_INT16") {
                settings.disable_int16 = true;
            }
            if vars.contains("DISABLE_INT8") {
                settings.disable_int8 = true;
            }
            if vars.contains("DISABLE_FLOAT64") {
                settings.disable_float64 = true;
            }
            if vars.contains("DISABLE_FLOAT16") {
                settings.disable_float16 = true;
            }

            settings
        }
        _ => EnvSettings::default(),
    }
});

#[derive(Debug, Default)]
pub struct EnvSettings {
    pub disable_int64: bool,
    pub disable_int16: bool,
    pub disable_int8: bool,
    pub disable_float64: bool,
    pub disable_float16: bool,
}

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

impl VultenDataType {
    pub fn size_of(&self) -> Result<usize, &'static str> {
        match *self {
            DT_FLOAT | DT_INT32 | DT_UINT32 => Ok(4),
            DT_INT64 | DT_UINT64 => Ok(8),
            _ => Err("Unknow type"),
        }
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

        look_for_features(&inst, &physical_device);
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default().maintenance4(true);
        let feat = vk::PhysicalDeviceFeatures::default().shader_int64(!ENV_SETTINGS.disable_int64);
        let feat2 = vk::PhysicalDeviceFeatures2::default()
            .features(feat)
            .push_next(&mut maintenance4);

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
                let q_pool_info = CommandPoolCreateInfo::default().queue_family_index(i as u32);
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

        let mut allocator_create_info =
            vk_mem::AllocatorCreateInfo::new(&inst, &device, physical_device);
        allocator_create_info.vulkan_api_version = VK_API_VER;
        allocator_create_info.flags = allocator_flags;
        let allocator = unsafe { vk_mem::Allocator::new(allocator_create_info) }
            .expect("Error could no create allocator!");

        let pipeline_cache_info = PipelineCacheCreateInfo::default();
        let pipeline_cache =
            unsafe { device.create_pipeline_cache(&pipeline_cache_info, None) }.unwrap();

        let mut sub_props = PhysicalDeviceSubgroupProperties::default();
        let mut props = PhysicalDeviceProperties2::default().push_next(&mut sub_props);
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
            pipelines: HashMap::new().into(),
            descriptor_pools: ArcSwap::from_pointee(Vec::new()),
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

        let instance_info = vk::InstanceCreateInfo::default().application_info(&appinfo);

        let inst = unsafe { entry.create_instance(&instance_info, None) }
            .expect("Error failed to create vkInstance!");

        unsafe { inst.enumerate_physical_devices() }
            .expect("Error no device")
            .len() as i32
    }

    pub fn get_device_name(&self) -> String {
        unsafe {
            CStr::from_ptr(
                self.vk_instance
                    .get_physical_device_properties(self.physical_device)
                    .device_name
                    .as_ptr(),
            )
            .to_string_lossy()
            .to_string()
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

    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&appinfo)
        .enabled_layer_names(&layers);

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

fn look_for_features(inst: &Instance, dev: &PhysicalDevice) {
    let mut maintenance4 = vk::PhysicalDeviceMaintenance4Features::default();
    let feat = vk::PhysicalDeviceFeatures::default();
    let mut feat2 = vk::PhysicalDeviceFeatures2::default()
        .features(feat)
        .push_next(&mut maintenance4);
    unsafe { inst.get_physical_device_features2(*dev, &mut feat2) };
    let maintenance4_feat =
        unsafe { *(feat2.p_next as *mut vk::PhysicalDeviceMaintenance4Features) };

    let type_error = |feat_name: &'static str, env_var: &'static str| {
        panic!("Reqested feature for type not present {}. Add {} to VULTEN_SETTINGS env var to disable it.", feat_name, env_var);
    };

    if maintenance4_feat.maintenance4 == 0 {
        panic!("Reqested feature not present Maintenance4");
    }
    if !ENV_SETTINGS.disable_int64 && feat2.features.shader_int64 == 0 {
        type_error("Int64", "DISABLE_INT64");
    }
}
