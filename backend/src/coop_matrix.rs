use ash::{vk, Entry, Instance};

#[derive(Debug)]
pub struct CoopMatrixFeature{
}

impl CoopMatrixFeature {
    pub fn new(entry: &Entry, inst: &Instance, device: &vk::PhysicalDevice) -> Self{
        let coop_inst = ash::khr::cooperative_matrix::Instance::new(&entry, &inst);
        let props = unsafe {coop_inst.get_physical_device_cooperative_matrix_properties(*device)}.unwrap();
        for prop in props{
            println!("{:?}", prop);
        }
        
        Self {  }
    }
}