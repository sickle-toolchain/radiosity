use ash::{khr::acceleration_structure, prelude::VkResult, vk};

use crate::vulkan::VkContext;

use super::Buffer;

pub struct AccelerationStructure<'a> {
    pub(crate) device: &'a acceleration_structure::Device,
    pub(crate) inner: vk::AccelerationStructureKHR,
    buffer: Buffer<'a>,
}

impl<'a> AccelerationStructure<'a> {
    pub fn build(
        vk_ctx @ &VkContext {
            device,
            queue,
            command_pool,
            ..
        }: &'a VkContext<'_>,
        as_device: &'a acceleration_structure::Device,
        ty: vk::AccelerationStructureTypeKHR,
        geometries: &[vk::AccelerationStructureGeometryKHR],
        ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
        flags: vk::BuildAccelerationStructureFlagsKHR,
    ) -> VkResult<Self> {
        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .flags(flags)
            .geometries(geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(ty);

        let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            as_device.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                ranges
                    .iter()
                    .map(|r| r.primitive_count)
                    .collect::<Vec<_>>()
                    .as_slice(),
                &mut sizes_info,
            );
        };

        let buffer = Buffer::new(
            vk_ctx,
            sizes_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(build_info.ty)
            .size(sizes_info.acceleration_structure_size)
            .buffer(buffer.handle())
            .offset(0);

        let inner = unsafe { as_device.create_acceleration_structure(&create_info, None) }?;
        build_info.dst_acceleration_structure = inner;

        let scratch_buffer = Buffer::new(
            vk_ctx,
            sizes_info.acceleration_structure_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_address(),
        };

        let build_command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(*command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers =
                unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap();
            command_buffers[0]
        };

        unsafe {
            device
                .begin_command_buffer(
                    build_command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            as_device.cmd_build_acceleration_structures(
                build_command_buffer,
                &[build_info],
                &[ranges],
            );
            device.end_command_buffer(build_command_buffer).unwrap();
            device.queue_submit(
                *queue,
                &[vk::SubmitInfo::default().command_buffers(&[build_command_buffer])],
                vk::Fence::null(),
            )?;

            device.queue_wait_idle(*queue).unwrap();
            device.free_command_buffers(*command_pool, &[build_command_buffer]);
            scratch_buffer.destroy();
        }

        Ok(Self {
            device: as_device,
            inner,
            buffer,
        })
    }

    pub fn handle(&self) -> vk::AccelerationStructureKHR {
        self.inner
    }

    pub fn device_address(&self) -> u64 {
        unsafe {
            self.device.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(self.inner),
            )
        }
    }

    pub fn destroy(self) {
        unsafe {
            self.device.destroy_acceleration_structure(self.inner, None);
            self.buffer.destroy();
        }
    }
}
