mod acceleration_structure;
mod buffer;

use std::ffi::CStr;

use ash::{prelude::VkResult, vk};

pub use acceleration_structure::AccelerationStructure;
pub use buffer::Buffer;

pub struct VkContext<'a> {
    device: &'a ash::Device,
    queue: &'a vk::Queue,
    command_pool: &'a vk::CommandPool,
    device_memory_properties: &'a vk::PhysicalDeviceMemoryProperties,
}

impl<'a> VkContext<'a> {
    pub fn new(
        device: &'a ash::Device,
        queue: &'a vk::Queue,
        command_pool: &'a vk::CommandPool,
        device_memory_properties: &'a vk::PhysicalDeviceMemoryProperties,
    ) -> VkResult<Self> {
        Ok(Self {
            device,
            queue,
            command_pool,
            device_memory_properties,
        })
    }
}

pub trait ApplicationInfoExt {
    fn application_from_env(self) -> Self;
}

impl ApplicationInfoExt for vk::ApplicationInfo<'_> {
    fn application_from_env(self) -> Self {
        let application_name =
            CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_NAME"), "\0").as_bytes())
                .expect("invalid package name");

        let major = env!("CARGO_PKG_VERSION_MAJOR")
            .parse::<u32>()
            .expect("invalid major version");
        let minor = env!("CARGO_PKG_VERSION_MINOR")
            .parse::<u32>()
            .expect("invalid minor version");
        let patch = env!("CARGO_PKG_VERSION_PATCH")
            .parse::<u32>()
            .expect("invalid patch version");

        self.application_name(application_name)
            .application_version(vk::make_api_version(0, major, minor, patch))
    }
}

pub trait PhysicalDeviceMemoryPropertiesExt {
    fn mem_ty_idx(
        &self,
        required_bits: u32,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32>;
}

impl PhysicalDeviceMemoryPropertiesExt for vk::PhysicalDeviceMemoryProperties {
    fn mem_ty_idx(
        &self,
        required_bits: u32,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for idx in 0..self.memory_type_count {
            let memory_properties = self.memory_types[idx as usize].property_flags;

            if (required_bits & (1 << idx)) == 1
                && (memory_properties & required_properties) == required_properties
            {
                return Some(idx);
            }
        }

        None
    }
}
