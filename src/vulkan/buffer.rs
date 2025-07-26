use std::ffi::c_void;

use crate::vulkan::VkContext;

use super::PhysicalDeviceMemoryPropertiesExt;
use ash::{prelude::VkResult, util::Align, vk};

pub struct Buffer {
    pub(crate) inner: vk::Buffer,
    pub(crate) device_memory: vk::DeviceMemory,
}

impl Buffer {
    pub fn new(
        VkContext {
            device,
            device_memory_properties,
            ..
        }: &VkContext<'_>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> VkResult<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let inner = unsafe { device.create_buffer(&buffer_info, None) }?;

        let memory_req = unsafe { device.get_buffer_memory_requirements(inner) };

        let memory_index = device_memory_properties
            .mem_ty_idx(memory_req.memory_type_bits, memory_properties)
            .expect("Failed to get memory type");

        let mut allocate_info_builder = vk::MemoryAllocateInfo::default();

        let mut memory_allocate_flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            allocate_info_builder =
                allocate_info_builder.push_next(&mut memory_allocate_flags_info);
        }

        let allocate_info = allocate_info_builder
            .allocation_size(memory_req.size)
            .memory_type_index(memory_index);

        let device_memory = unsafe { device.allocate_memory(&allocate_info, None) }?;

        unsafe { device.bind_buffer_memory(inner, device_memory, 0) }?;

        Ok(Self {
            inner,
            device_memory,
        })
    }

    pub fn handle(&self) -> vk::Buffer {
        self.inner
    }

    pub fn device_address(&self, device: &ash::Device) -> u64 {
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(self.inner);
        unsafe { device.get_buffer_device_address(&address_info) }
    }

    pub fn store<T: Copy>(&mut self, data: &[T], device: &ash::Device) {
        unsafe {
            let size = std::mem::size_of_val(data) as u64;
            let mapped_ptr = self.map(size, device);
            let mut mapped_slice = Align::new(mapped_ptr, std::mem::align_of::<T>() as u64, size);
            mapped_slice.copy_from_slice(data);
            self.unmap(device);
        }
    }

    fn map(&mut self, size: vk::DeviceSize, device: &ash::Device) -> *mut c_void {
        unsafe {
            let data: *mut std::ffi::c_void = device
                .map_memory(self.device_memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            data
        }
    }

    fn unmap(&mut self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.device_memory);
        }
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.inner, None);
            device.free_memory(self.device_memory, None);
        }
    }
}
