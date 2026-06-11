//! Vulkan ray-tracing application

use std::ptr;
use std::rc::Rc;

use anyhow::{Context, Result};
use tracing::{info_span, instrument};

use ash::khr::{acceleration_structure, ray_tracing_pipeline};
use ash::vk::{
    self, AccelerationStructureBuildRangeInfoKHR, AccelerationStructureGeometryKHR,
    AccelerationStructureKHR, AccelerationStructureTypeKHR, BufferUsageFlags,
    BuildAccelerationStructureFlagsKHR, MemoryPropertyFlags, Packed24_8,
    PhysicalDeviceRayTracingPipelinePropertiesKHR, QueryPool,
};

use crate::vulkan::{Buffer, GeometryIndex, GeometryVertex, VulkanContext};

#[repr(align(4))]
struct AlignedSpirv<T: ?Sized>(T);

static SHADER: &AlignedSpirv<[u8]> =
    &AlignedSpirv(*include_bytes!(env!("radiosity_shader.spv")));

#[repr(u32)]
#[derive(Copy, Clone)]
enum TimestampSlot {
    SkyBegin = 0,
    SkyEnd,
    WorldBegin,
    WorldEnd,
    GiBegin,
    GiEnd,
}

impl TimestampSlot {
    const COUNT: u32 = Self::GiEnd as u32 + 1;
}

pub struct Application<'a> {
    pub ctx: Rc<VulkanContext>,
    pub acceleration_structure_device: acceleration_structure::Device,
    pub ray_tracing_pipeline_device: ray_tracing_pipeline::Device,
    pub ray_tracing_pipeline_properties: PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,

    pub scratch_offset_alignment: u32,

    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,

    pub blas: AccelerationStructureKHR,
    pub blas_buffer: Option<Buffer>,
    pub sky_blas: AccelerationStructureKHR,
    pub sky_blas_buffer: Option<Buffer>,
    pub tlas: AccelerationStructureKHR,
    pub tlas_buffer: Option<Buffer>,

    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub shader_binding_table_buffer: Option<Buffer>,

    pub command_buffer: vk::CommandBuffer,

    pub timestamp_query_pool: QueryPool,
}

impl Application<'_> {
    pub const SBT_GROUP_RAYGEN_SKY: u32 = 0;
    pub const SBT_GROUP_RAYGEN_WORLD: u32 = 1;
    pub const SBT_GROUP_RAYGEN_GI: u32 = 2;
    pub const SBT_GROUP_MISS: u32 = 3;
    pub const SBT_GROUP_HIT: u32 = 4;
    pub const SBT_GROUP_SKY_HIT: u32 = 5;

    pub fn new(ctx: Rc<VulkanContext>) -> Result<Self> {
        let acceleration_structure_device =
            acceleration_structure::Device::new(&ctx.instance, &ctx.device);
        let ray_tracing_pipeline_device =
            ray_tracing_pipeline::Device::new(&ctx.instance, &ctx.device);

        let mut ray_tracing_pipeline_properties =
            PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut acceleration_structure_properties =
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        {
            let mut physical_device_properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut ray_tracing_pipeline_properties)
                .push_next(&mut acceleration_structure_properties);

            unsafe {
                ctx.instance.get_physical_device_properties2(
                    ctx.physical_device,
                    &mut physical_device_properties2,
                );
            }
        }
        let scratch_offset_alignment =
            acceleration_structure_properties.min_acceleration_structure_scratch_offset_alignment;

        let timestamp_query_pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TimestampSlot::COUNT);

        let timestamp_query_pool = unsafe {
            ctx.device
                .create_query_pool(&timestamp_query_pool_info, None)?
        };

        let command_buffer = {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(ctx.pool)
                .level(vk::CommandBufferLevel::PRIMARY);
            unsafe { ctx.device.allocate_command_buffers(&info) }?[0]
        };

        Ok(Self {
            ctx,
            acceleration_structure_device,
            ray_tracing_pipeline_device,
            ray_tracing_pipeline_properties,
            scratch_offset_alignment,
            timestamp_query_pool,
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            blas: AccelerationStructureKHR::null(),
            blas_buffer: None,
            sky_blas: AccelerationStructureKHR::null(),
            sky_blas_buffer: None,
            tlas: AccelerationStructureKHR::null(),
            tlas_buffer: None,

            vertex_buffer: None,
            index_buffer: None,

            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            shader_binding_table_buffer: None,
            command_buffer,
        })
    }

    fn reset_timestamp_pool(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.ctx.device.cmd_reset_query_pool(
                command_buffer,
                self.timestamp_query_pool,
                0,
                TimestampSlot::COUNT,
            );
        }
    }

    fn write_timestamp(
        &self,
        command_buffer: vk::CommandBuffer,
        slot: TimestampSlot,
        stage: vk::PipelineStageFlags2,
    ) {
        unsafe {
            self.ctx.device.cmd_write_timestamp2(
                command_buffer,
                stage,
                self.timestamp_query_pool,
                slot as u32,
            );
        }
    }

    fn wait_timestamp(&self, slot: TimestampSlot) -> Result<()> {
        let mut ts = [0u64; 1];
        unsafe {
            self.ctx.device.get_query_pool_results(
                self.timestamp_query_pool,
                slot as u32,
                &mut ts,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        Ok(())
    }

    fn elapsed_ns(&self, start: TimestampSlot, end: TimestampSlot) -> Result<f64> {
        let mut start_ts = [0u64; 1];
        let mut end_ts = [0u64; 1];
        unsafe {
            self.ctx.device.get_query_pool_results(
                self.timestamp_query_pool,
                start as u32,
                &mut start_ts,
                vk::QueryResultFlags::TYPE_64,
            )?;
            self.ctx.device.get_query_pool_results(
                self.timestamp_query_pool,
                end as u32,
                &mut end_ts,
                vk::QueryResultFlags::TYPE_64,
            )?;
        }
        let valid_bits = self.ctx.timestamp_valid_bits;
        let mask = if valid_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << valid_bits) - 1
        };
        let delta_ticks = (end_ts[0] & mask).wrapping_sub(start_ts[0] & mask) & mask;
        Ok(delta_ticks as f64 * self.ctx.physical_device_properties.limits.timestamp_period as f64)
    }

    #[instrument(skip_all)]
    fn create_triangle_geometry<V, I>(
        &'_ self,
        vertex_buffer: &Buffer,
        index_buffer: &Buffer,
        vertex_count: u32,
    ) -> vk::AccelerationStructureGeometryKHR<'_>
    where
        V: GeometryVertex + Copy,
        I: GeometryIndex + Copy,
    {
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: vertex_buffer.device_address(),
                    })
                    .max_vertex(vertex_count.saturating_sub(1))
                    .vertex_stride(V::vk_stride())
                    .vertex_format(V::vk_format())
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: index_buffer.device_address(),
                    })
                    .index_type(I::vk_index_type()),
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE)
    }

    #[instrument(skip_all)]
    fn create_input_buffer<T: Copy>(
        &self,
        command_buffer: vk::CommandBuffer,
        staging: &mut Vec<Buffer>,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer> {
        let bytes = size_of_val(data) as vk::DeviceSize;

        let mut src = Buffer::new(
            self.ctx.clone(),
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        src.store(data);

        let dst = Buffer::new(
            self.ctx.clone(),
            bytes,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::TRANSFER_DST
                | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        dst.cmd_copy_from(command_buffer, &src, bytes);

        staging.push(src);
        Ok(dst)
    }

    // Create instance acceleration structure geometry
    #[instrument(skip_all)]
    fn create_instance_geometry(
        &'_ self,
        instances: &[vk::AccelerationStructureInstanceKHR],
    ) -> Result<(AccelerationStructureGeometryKHR<'_>, Buffer)> {
        const INPUT_BUFFER_FLAGS: BufferUsageFlags = BufferUsageFlags::from_raw(
            BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw()
                | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR.as_raw(),
        );
        const MEMORY_PROPERTIES: MemoryPropertyFlags = MemoryPropertyFlags::from_raw(
            MemoryPropertyFlags::HOST_VISIBLE.as_raw()
                | MemoryPropertyFlags::HOST_COHERENT.as_raw(),
        );

        let mut instance_buffer = Buffer::new(
            self.ctx.clone(),
            size_of_val(instances) as vk::DeviceSize,
            INPUT_BUFFER_FLAGS,
            MEMORY_PROPERTIES,
        )?;
        instance_buffer.store(instances);

        let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer.device_address(),
            });

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances });

        Ok((geometry, instance_buffer))
    }

    // Create acceleration structure
    #[instrument(skip(self, geometries, build_range_info, flags))]
    fn create_as(
        &self,
        command_buffer: vk::CommandBuffer,
        structure_type: AccelerationStructureTypeKHR,
        geometries: &[AccelerationStructureGeometryKHR],
        build_range_info: &[AccelerationStructureBuildRangeInfoKHR],
        flags: BuildAccelerationStructureFlagsKHR,
    ) -> Result<(AccelerationStructureKHR, Buffer, Buffer)> {
        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .flags(flags)
            .geometries(geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(structure_type);

        let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.acceleration_structure_device
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    build_range_info
                        .iter()
                        .map(|r| r.primitive_count)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    &mut sizes_info,
                );
        };

        let buffer = Buffer::new(
            self.ctx.clone(),
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

        let acceleration_structure = unsafe {
            self.acceleration_structure_device
                .create_acceleration_structure(&create_info, None)
        }?;
        build_info.dst_acceleration_structure = acceleration_structure;

        let scratch_alignment = self.scratch_offset_alignment.max(1) as vk::DeviceSize;
        let scratch_buffer = Buffer::new(
            self.ctx.clone(),
            sizes_info.build_scratch_size + scratch_alignment - 1,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let scratch_address =
            (scratch_buffer.device_address() + scratch_alignment - 1) & !(scratch_alignment - 1);

        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_address,
        };

        unsafe {
            self.acceleration_structure_device
                .cmd_build_acceleration_structures(
                    command_buffer,
                    &[build_info],
                    &[build_range_info],
                );
        }

        Ok((acceleration_structure, buffer, scratch_buffer))
    }

    #[instrument(skip_all)]
    pub fn create_acceleration_structures<V, I>(
        &mut self,
        command_buffer: vk::CommandBuffer,
        vertices: &[V],
        solid_indices: &[I],
        sky_indices: &[I],
    ) -> Result<Vec<Buffer>>
    where
        V: GeometryVertex + Copy,
        I: GeometryIndex + Copy,
    {
        let mut temp_buffers = Vec::new();

        let vertex_buffer = self.create_input_buffer(
            command_buffer,
            &mut temp_buffers,
            vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let solid_index_buffer = self.create_input_buffer(
            command_buffer,
            &mut temp_buffers,
            solid_indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        let sky_index_buffer = if sky_indices.is_empty() {
            None
        } else {
            Some(self.create_input_buffer(
                command_buffer,
                &mut temp_buffers,
                sky_indices,
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?)
        };

        let upload_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
            );
        unsafe {
            self.ctx.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&[upload_barrier]),
            );
        }

        let solid_geometry = self.create_triangle_geometry::<V, I>(
            &vertex_buffer,
            &solid_index_buffer,
            vertices.len() as u32,
        );

        let (solid_blas, solid_blas_buffer, solid_scratch) = self.create_as(
            command_buffer,
            AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &[solid_geometry],
            &[vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(solid_indices.len() as u32 / 3)],
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
        )?;
        temp_buffers.push(solid_scratch);

        // Build sky BLAS if there are sky faces
        let sky_blas_opt = if let Some(sky_index_buffer) = sky_index_buffer {
            let sky_geometry = self.create_triangle_geometry::<V, I>(
                &vertex_buffer,
                &sky_index_buffer,
                vertices.len() as u32,
            );
            let (sky_blas, sky_blas_buffer, sky_scratch) = self.create_as(
                command_buffer,
                AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                &[sky_geometry],
                &[vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(sky_indices.len() as u32 / 3)],
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            )?;
            temp_buffers.push(sky_scratch);
            temp_buffers.push(sky_index_buffer);
            Some((sky_blas, sky_blas_buffer))
        } else {
            None
        };

        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR);
        unsafe {
            self.ctx.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().memory_barriers(&[barrier]),
            );
        }

        let identity_matrix: [f32; 12] =
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let solid_device_handle = unsafe {
            self.acceleration_structure_device
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(solid_blas),
                )
        };

        let mut instances = vec![vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: identity_matrix,
            },
            instance_custom_index_and_mask: Packed24_8::new(0, shared::MASK_SOLID),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: solid_device_handle,
            },
        }];

        if let Some((sky_blas, _)) = &sky_blas_opt {
            let sky_device_handle = unsafe {
                self.acceleration_structure_device
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(*sky_blas),
                    )
            };
            instances.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: identity_matrix,
                },
                instance_custom_index_and_mask: Packed24_8::new(0, shared::MASK_SKY),
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                    1,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: sky_device_handle,
                },
            });
        }

        let (tlas_geometry, tlas_geometry_buffer) = self.create_instance_geometry(&instances)?;
        temp_buffers.push(tlas_geometry_buffer);

        let (tlas, tlas_buffer, tlas_scratch) = self.create_as(
            command_buffer,
            AccelerationStructureTypeKHR::TOP_LEVEL,
            &[tlas_geometry],
            &[vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(instances.len() as u32)],
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        )?;
        temp_buffers.push(tlas_scratch);

        self.blas = solid_blas;
        self.blas_buffer = Some(solid_blas_buffer);
        if let Some((sky_blas, sky_blas_buffer)) = sky_blas_opt {
            self.sky_blas = sky_blas;
            self.sky_blas_buffer = Some(sky_blas_buffer);
        }
        self.tlas = tlas;
        self.tlas_buffer = Some(tlas_buffer);

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(solid_index_buffer);
        Ok(temp_buffers)
    }

    #[instrument(skip_all)]
    pub fn create_pipeline(&mut self) -> Result<usize> {
        self.descriptor_set_layout = unsafe {
            self.ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    vk::DescriptorSetLayoutBinding::default()
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                        .binding(0),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(3)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(4)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                ]),
                None,
            )
        }?;

        let layouts = vec![self.descriptor_set_layout];
        let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

        self.pipeline_layout = unsafe {
            self.ctx
                .device
                .create_pipeline_layout(&layout_create_info, None)
        }?;

        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: SHADER.0.len(),
            p_code: SHADER.0.as_ptr().cast(),
            ..Default::default()
        };

        let shader_module = unsafe {
            self.ctx
                .device
                .create_shader_module(&shader_module_create_info, None)
        }?;

        let general_group = |shader: u32| {
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(shader)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
        };
        let hit_group = |closest_hit: u32| {
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(closest_hit)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
        };

        let shader_groups = vec![
            general_group(Self::SBT_GROUP_RAYGEN_SKY),
            general_group(Self::SBT_GROUP_RAYGEN_WORLD),
            general_group(Self::SBT_GROUP_RAYGEN_GI),
            general_group(Self::SBT_GROUP_MISS),
            hit_group(Self::SBT_GROUP_HIT),
            hit_group(Self::SBT_GROUP_SKY_HIT),
        ];

        let shader_stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(shader_module)
                .name(c"ray_generation_sky"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(shader_module)
                .name(c"ray_generation_world"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(shader_module)
                .name(c"ray_generation_gi"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(shader_module)
                .name(c"miss"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(shader_module)
                .name(c"closest_hit"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(shader_module)
                .name(c"sky_hit"),
        ];

        let pipeline_result = unsafe {
            self.ray_tracing_pipeline_device
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[vk::RayTracingPipelineCreateInfoKHR::default()
                        .stages(&shader_stages)
                        .groups(&shader_groups)
                        .max_pipeline_ray_recursion_depth(1)
                        .layout(self.pipeline_layout)],
                    None,
                )
        };

        unsafe {
            self.ctx.device.destroy_shader_module(shader_module, None);
        }

        self.pipeline = pipeline_result.map_err(|(_, result)| result)?[0];

        Ok(shader_groups.len())
    }

    #[instrument(skip_all)]
    pub fn create_shader_binding_table(&mut self, shader_group_count: usize) -> Result<u64> {
        let handle_size = self
            .ray_tracing_pipeline_properties
            .shader_group_handle_size as usize;
        let handle_alignment = self
            .ray_tracing_pipeline_properties
            .shader_group_base_alignment as usize;
        let handle_size_aligned = (handle_size + handle_alignment - 1) & !(handle_alignment - 1);

        let incoming_table_data = unsafe {
            self.ray_tracing_pipeline_device
                .get_ray_tracing_shader_group_handles(
                    self.pipeline,
                    0,
                    shader_group_count as u32,
                    shader_group_count * handle_size,
                )
        }?;

        let table_size = shader_group_count * handle_size_aligned;
        let mut table_data = vec![0u8; table_size];

        for i in 0..shader_group_count {
            let src = i * handle_size;
            let dst = i * handle_size_aligned;
            table_data[dst..dst + handle_size]
                .copy_from_slice(&incoming_table_data[src..src + handle_size]);
        }

        let mut shader_binding_table_buffer = Buffer::new(
            self.ctx.clone(),
            table_size as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        shader_binding_table_buffer.store(&table_data);

        self.shader_binding_table_buffer = Some(shader_binding_table_buffer);

        Ok(handle_size_aligned as u64)
    }

    #[instrument(skip_all)]
    pub fn create_descriptor_set(
        &mut self,
        texel_buffer: &Buffer,
        output_buffer: &Buffer,
        sky_buffer: &Buffer,
        world_lights_buffer: &Buffer,
    ) -> Result<()> {
        let descriptor_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4,
            },
        ];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_sizes)
            .max_sets(1);

        let descriptor_pool = unsafe {
            self.ctx
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }?;

        let descriptor_set = unsafe {
            self.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[self.descriptor_set_layout]),
            )
        }?[0];

        let accel_structs = [self.tlas];
        let mut accel_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&accel_structs);

        let accel_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut accel_info)
            .descriptor_count(1);

        let texel_info = [vk::DescriptorBufferInfo::default()
            .buffer(texel_buffer.handle())
            .range(vk::WHOLE_SIZE)];

        let texel_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&texel_info)
            .descriptor_count(1);

        let output_info = [vk::DescriptorBufferInfo::default()
            .buffer(output_buffer.handle())
            .range(vk::WHOLE_SIZE)];

        let output_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&output_info)
            .descriptor_count(1);

        let world_info = [vk::DescriptorBufferInfo::default()
            .buffer(world_lights_buffer.handle())
            .range(vk::WHOLE_SIZE)];

        let world_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&world_info)
            .descriptor_count(1);

        let sky_info = [vk::DescriptorBufferInfo::default()
            .buffer(sky_buffer.handle())
            .range(vk::WHOLE_SIZE)];

        let sky_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(4)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&sky_info)
            .descriptor_count(1);

        unsafe {
            self.ctx.device.update_descriptor_sets(
                &[
                    accel_write,
                    texel_write,
                    output_write,
                    world_write,
                    sky_write,
                ],
                &[],
            );
        }

        self.descriptor_pool = descriptor_pool;
        self.descriptor_set = descriptor_set;

        Ok(())
    }

    #[instrument(skip_all)]
    pub fn record_ray_tracing(
        &self,
        sbt_handle_size: u64,
        texel_count: usize,
        output_device: &Buffer,
        output_readback: Option<&Buffer>,
    ) -> Result<()> {
        let sbt_buffer = self
            .shader_binding_table_buffer
            .as_ref()
            .context("SBT buffer not created")?;

        let sbt_address = sbt_buffer.device_address();

        let sbt_raygen_region_sky = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + Self::SBT_GROUP_RAYGEN_SKY as u64 * sbt_handle_size)
            .size(sbt_handle_size)
            .stride(sbt_handle_size);

        let sbt_raygen_region_world = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + Self::SBT_GROUP_RAYGEN_WORLD as u64 * sbt_handle_size)
            .size(sbt_handle_size)
            .stride(sbt_handle_size);

        let sbt_raygen_region_gi = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + Self::SBT_GROUP_RAYGEN_GI as u64 * sbt_handle_size)
            .size(sbt_handle_size)
            .stride(sbt_handle_size);

        let sbt_miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + Self::SBT_GROUP_MISS as u64 * sbt_handle_size)
            .size(sbt_handle_size)
            .stride(sbt_handle_size);

        let sbt_hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + Self::SBT_GROUP_HIT as u64 * sbt_handle_size)
            .size(2 * sbt_handle_size)
            .stride(sbt_handle_size);

        let sbt_call_region = vk::StridedDeviceAddressRegionKHR::default();

        let gpu_span = info_span!("GPU");
        let rt_stage = vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        let entered = gpu_span.enter();

        let sky_span = info_span!(
            "PASS: Sky Illumination",
            elapsed_ns = ::tracing::field::Empty
        );
        let world_span = info_span!(
            "PASS: World Illumination",
            elapsed_ns = ::tracing::field::Empty
        );
        let gi_span = info_span!(
            "PASS: Global Illumination",
            elapsed_ns = ::tracing::field::Empty
        );
        unsafe {
            self.reset_timestamp_pool(self.command_buffer);

            self.ctx.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );
            self.ctx.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            self.write_timestamp(self.command_buffer, TimestampSlot::SkyBegin, rt_stage);
            self.ray_tracing_pipeline_device.cmd_trace_rays(
                self.command_buffer,
                &sbt_raygen_region_sky,
                &sbt_miss_region,
                &sbt_hit_region,
                &sbt_call_region,
                texel_count as u32,
                1,
                1,
            );
            self.write_timestamp(self.command_buffer, TimestampSlot::SkyEnd, rt_stage);

            let barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(output_device.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE);
            self.ctx.device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[barrier]),
            );

            self.write_timestamp(self.command_buffer, TimestampSlot::WorldBegin, rt_stage);
            self.ray_tracing_pipeline_device.cmd_trace_rays(
                self.command_buffer,
                &sbt_raygen_region_world,
                &sbt_miss_region,
                &sbt_hit_region,
                &sbt_call_region,
                texel_count as u32,
                1,
                1,
            );
            self.write_timestamp(self.command_buffer, TimestampSlot::WorldEnd, rt_stage);

            self.write_timestamp(self.command_buffer, TimestampSlot::GiBegin, rt_stage);
            let bounce_count = 1;
            for _bounce in 0..bounce_count {
                let barrier = vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(output_device.handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);

                self.ctx.device.cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&[barrier]),
                );

                self.ray_tracing_pipeline_device.cmd_trace_rays(
                    self.command_buffer,
                    &sbt_raygen_region_gi,
                    &sbt_miss_region,
                    &sbt_hit_region,
                    &sbt_call_region,
                    texel_count as u32,
                    1,
                    1,
                );
            }
            self.write_timestamp(self.command_buffer, TimestampSlot::GiEnd, rt_stage);

            if let Some(readback) = output_readback {
                let to_transfer = vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(output_device.handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                self.ctx.device.cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&[to_transfer]),
                );

                self.ctx.device.cmd_copy_buffer(
                    self.command_buffer,
                    output_device.handle(),
                    readback.handle(),
                    &[vk::BufferCopy::default().size(readback.size())],
                );

                let to_host = vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(readback.handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                self.ctx.device.cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&[to_host]),
                );
            } else {
                let to_shader = vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(output_device.handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                self.ctx.device.cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&[to_shader]),
                );
            }

            self.ctx.device.end_command_buffer(self.command_buffer)?;
            self.ctx.device.queue_submit(
                self.ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.command_buffer])],
                vk::Fence::null(),
            )?;
        }

        {
            let _e = sky_span.enter();
            self.wait_timestamp(TimestampSlot::SkyEnd)?;
        }
        sky_span.record(
            "elapsed_ns",
            self.elapsed_ns(TimestampSlot::SkyBegin, TimestampSlot::SkyEnd)?,
        );

        {
            let _e = world_span.enter();
            self.wait_timestamp(TimestampSlot::WorldEnd)?;
        }
        world_span.record(
            "elapsed_ns",
            self.elapsed_ns(TimestampSlot::WorldBegin, TimestampSlot::WorldEnd)?,
        );

        {
            let _e = gi_span.enter();
            self.wait_timestamp(TimestampSlot::GiEnd)?;
        }
        gi_span.record(
            "elapsed_ns",
            self.elapsed_ns(TimestampSlot::GiBegin, TimestampSlot::GiEnd)?,
        );

        unsafe { self.ctx.device.queue_wait_idle(self.ctx.queue)? };

        drop(entered);
        Ok(())
    }
}

impl Drop for Application<'_> {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.device_wait_idle().ok();
            self.ctx.device.destroy_pipeline(self.pipeline, None);

            self.ctx
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.ctx
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.ctx
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.acceleration_structure_device
                .destroy_acceleration_structure(self.tlas, None);

            self.acceleration_structure_device
                .destroy_acceleration_structure(self.blas, None);
            if self.sky_blas != AccelerationStructureKHR::null() {
                self.acceleration_structure_device
                    .destroy_acceleration_structure(self.sky_blas, None);
            }
            self.ctx
                .device
                .free_command_buffers(self.ctx.pool, &[self.command_buffer]);
            self.ctx
                .device
                .destroy_query_pool(self.timestamp_query_pool, None);
        }
    }
}
