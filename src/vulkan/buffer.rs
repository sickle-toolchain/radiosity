use anyhow::Result;
use std::ffi::c_void;
use std::rc::Rc;

use log::debug;

use ash::prelude::VkResult;
use ash::util::Align;
use ash::vk::{self, Handle};

use super::PhysicalDeviceMemoryPropertiesExt;
use crate::vulkan::VulkanContext;

pub struct Buffer {
    pub(crate) ctx: Rc<VulkanContext>,
    pub(crate) inner: vk::Buffer,
    pub(crate) device_memory: vk::DeviceMemory,
    pub(crate) size: vk::DeviceSize,
}

impl Buffer {
    pub fn new(
        ctx: Rc<VulkanContext>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> VkResult<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let inner = unsafe { ctx.device.create_buffer(&buffer_info, None) }?;

        let memory_req = unsafe { ctx.device.get_buffer_memory_requirements(inner) };

        let memory_index = ctx
            .physical_device_memory_properties
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

        let device_memory = unsafe { ctx.device.allocate_memory(&allocate_info, None) }?;
        debug!(
            "Allocated 0x{:x} bytes @ 0x{:x}",
            allocate_info.allocation_size,
            device_memory.as_raw()
        );

        unsafe { ctx.device.bind_buffer_memory(inner, device_memory, 0) }?;

        Ok(Self {
            ctx,
            inner,
            device_memory,
            size,
        })
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn handle(&self) -> vk::Buffer {
        self.inner
    }

    pub fn device_address(&self) -> u64 {
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(self.inner);
        unsafe { self.ctx.device.get_buffer_device_address(&address_info) }
    }

    pub fn store<T: Copy>(&mut self, data: &[T]) {
        unsafe {
            let size = std::mem::size_of_val(data) as u64;
            let mapped_ptr = self.map(size);
            let mut mapped_slice = Align::new(mapped_ptr, std::mem::align_of::<T>() as u64, size);
            mapped_slice.copy_from_slice(data);
            self.unmap();
        }
    }

    pub fn load<T: Copy>(&self, element_count: usize) -> Vec<T> {
        unsafe {
            let size = (std::mem::size_of::<T>() * element_count) as u64;
            let mapped_ptr = self
                .ctx
                .device
                .map_memory(self.device_memory, 0, size, vk::MemoryMapFlags::empty())
                .expect("Failed to map memory") as *const T;

            let slice = std::slice::from_raw_parts(mapped_ptr, element_count);
            let result = slice.to_vec();

            self.ctx.device.unmap_memory(self.device_memory);

            result
        }
    }

    fn map(&mut self, size: vk::DeviceSize) -> *mut c_void {
        unsafe {
            let data: *mut std::ffi::c_void = self
                .ctx
                .device
                .map_memory(self.device_memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            data
        }
    }

    fn unmap(&mut self) {
        unsafe {
            self.ctx.device.unmap_memory(self.device_memory);
        }
    }

    pub fn destroy(&self) {
        unsafe {
            self.ctx.device.destroy_buffer(self.inner, None);
            self.ctx.device.free_memory(self.device_memory, None);
        }
    }

    pub fn copy_from(&self, src: &Buffer, size: vk::DeviceSize) -> Result<()> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(self.ctx.pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffer = unsafe { self.ctx.device.allocate_command_buffers(&allocate_info) }?[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.ctx
                .device
                .begin_command_buffer(command_buffer, &begin)?;
            self.ctx.device.cmd_copy_buffer(
                command_buffer,
                src.handle(),
                self.handle(),
                &[vk::BufferCopy::default().size(size)],
            );
            self.ctx.device.end_command_buffer(command_buffer)?;

            self.ctx.device.queue_submit(
                self.ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                vk::Fence::null(),
            )?;
            self.ctx.device.queue_wait_idle(self.ctx.queue)?;

            self.ctx
                .device
                .free_command_buffers(self.ctx.pool, &[command_buffer]);
        }

        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.destroy();
    }
}
