use std::borrow::Cow;
use std::cell::{Ref, RefMut};
use std::fs::File;
use std::path::PathBuf;
use std::ptr;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use tracing::{error, info, info_span, instrument, warn};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_timing::TreeTimingLayer;
use zerocopy::IntoBytes;

use ash::khr::{
    acceleration_structure, deferred_host_operations, get_memory_requirements2,
    ray_tracing_pipeline, spirv_1_4,
};
use ash::vk::{
    self, AccelerationStructureBuildRangeInfoKHR, AccelerationStructureGeometryKHR,
    AccelerationStructureKHR, AccelerationStructureTypeKHR, BufferUsageFlags,
    BuildAccelerationStructureFlagsKHR, MemoryPropertyFlags, Packed24_8,
    PhysicalDeviceRayTracingPipelinePropertiesKHR, QueryPool,
};
use spirv_std::glam::{Mat3, Vec3};

use bsp::Bsp;
use lump_definitions::source::{
    ColorRGBExp32, EmitType, Face, LumpDefinition, Plane, SurfaceEdge, SurfaceFlags, TextureInfo,
    Vertex, WorldLight,
};

use radiosity::Associated;
use radiosity::vulkan::{Buffer, GeometryIndex, GeometryVertex, VulkanContext};
use shared::{AlignedVec3, TexelData};

pub trait LumpData: zerocopy::FromBytes + zerocopy::KnownLayout + zerocopy::Immutable {}
impl<T: ?Sized + zerocopy::FromBytes + zerocopy::KnownLayout + zerocopy::Immutable> LumpData for T {}

pub trait MutableLumpData: LumpData + zerocopy::IntoBytes {}
impl<T: ?Sized + LumpData + zerocopy::IntoBytes> MutableLumpData for T {}

pub trait BspExt<'a> {
    fn get_lump<T>(&'a self, def: LumpDefinition) -> Result<Ref<'a, T>>
    where
        T: ?Sized + LumpData;

    fn get_lump_mut<T>(&'a self, def: LumpDefinition) -> Result<RefMut<'a, T>>
    where
        T: ?Sized + MutableLumpData;
}

impl<'a> BspExt<'a> for Bsp<'a> {
    fn get_lump<T>(&'a self, def: LumpDefinition) -> Result<Ref<'a, T>>
    where
        T: ?Sized + LumpData,
    {
        self.lump_cast::<T, _>(def)
            .map_err(|_| anyhow!("Failed to get lump"))
    }

    fn get_lump_mut<T>(&'a self, def: LumpDefinition) -> Result<RefMut<'a, T>>
    where
        T: ?Sized + MutableLumpData,
    {
        self.lump_cast_mut::<T, _>(def)
            .map_err(|_| anyhow!("Failed to get lump"))
    }
}

const SHADER: &[u8] = include_bytes!(env!("radiosity_shader.spv"));

pub struct Application<'a> {
    pub ctx: Rc<VulkanContext>,
    pub acceleration_structure_device: acceleration_structure::Device,
    pub ray_tracing_pipeline_device: ray_tracing_pipeline::Device,
    pub ray_tracing_pipeline_properties: PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,

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
    pub const SBT_GROUP_RAYGEN_DIRECT: u32 = 0;
    pub const SBT_GROUP_RAYGEN_GI: u32 = 1;
    pub const SBT_GROUP_MISS: u32 = 2;
    pub const SBT_GROUP_HIT: u32 = 3;
    pub const SBT_GROUP_SKY_HIT: u32 = 4;

    pub fn new(ctx: Rc<VulkanContext>) -> Result<Self> {
        let acceleration_structure_device =
            acceleration_structure::Device::new(&ctx.instance, &ctx.device);
        let ray_tracing_pipeline_device =
            ray_tracing_pipeline::Device::new(&ctx.instance, &ctx.device);

        let mut ray_tracing_pipeline_properties =
            PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        {
            let mut physical_device_properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut ray_tracing_pipeline_properties);

            unsafe {
                ctx.instance.get_physical_device_properties2(
                    ctx.physical_device,
                    &mut physical_device_properties2,
                );
            }
        }

        let timestamp_query_pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);

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

    fn begin_timestamp_range(
        &self,
        command_buffer: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
    ) {
        unsafe {
            self.ctx
                .device
                .cmd_reset_query_pool(command_buffer, self.timestamp_query_pool, 0, 2);
            self.ctx.device.cmd_write_timestamp(
                command_buffer,
                stage,
                self.timestamp_query_pool,
                0,
            );
        }
    }

    fn end_timestamp_range(
        &self,
        command_buffer: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
    ) {
        unsafe {
            self.ctx.device.cmd_write_timestamp(
                command_buffer,
                stage,
                self.timestamp_query_pool,
                1,
            );
        }
    }

    fn read_elapsed_ns(&self) -> Result<f64> {
        let mut timestamps = [0u64; 2];
        unsafe {
            self.ctx.device.get_query_pool_results(
                self.timestamp_query_pool,
                0,
                &mut timestamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let delta_ticks = timestamps[1].saturating_sub(timestamps[0]);
        let time_ns =
            delta_ticks as f64 * self.ctx.physical_device_properties.limits.timestamp_period as f64;
        Ok(time_ns)
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
        &'_ self,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer> {
        let mut buffer = Buffer::new(
            self.ctx.clone(),
            size_of_val(data) as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        buffer.store(data);
        Ok(buffer)
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

        let scratch_buffer = Buffer::new(
            self.ctx.clone(),
            sizes_info.build_scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_address(),
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
        let vertex_buffer =
            self.create_input_buffer(vertices, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let solid_index_buffer =
            self.create_input_buffer(solid_indices, vk::BufferUsageFlags::INDEX_BUFFER)?;

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
        let sky_blas_opt = if !sky_indices.is_empty() {
            let sky_index_buffer =
                self.create_input_buffer(sky_indices, vk::BufferUsageFlags::INDEX_BUFFER)?;
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

        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
        unsafe {
            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
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
            instance_custom_index_and_mask: Packed24_8::new(0, 0xFF),
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
                instance_custom_index_and_mask: Packed24_8::new(0, 0xFF),
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
        let binding_flags_inner = [
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
        ];

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
            .binding_flags(&binding_flags_inner);

        self.descriptor_set_layout = unsafe {
            self.ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&[
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
                    ])
                    .push_next(&mut binding_flags),
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
            code_size: SHADER.len(),
            p_code: SHADER.as_ptr().cast(),
            ..Default::default()
        };

        let shader_module = unsafe {
            self.ctx
                .device
                .create_shader_module(&shader_module_create_info, None)
        }?;

        let shader_groups = vec![
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(Self::SBT_GROUP_RAYGEN_DIRECT)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(Self::SBT_GROUP_RAYGEN_GI)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(Self::SBT_GROUP_MISS)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(Self::SBT_GROUP_HIT)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(Self::SBT_GROUP_SKY_HIT)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        let shader_stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(shader_module)
                .name(c"ray_generation_direct"),
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
        lighting_buffer: &Buffer,
        lights_buffer: &Buffer,
    ) -> Result<()> {
        let descriptor_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 3,
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

        let mut count_allocate_info =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(&[1]);

        let descriptor_set = unsafe {
            self.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[self.descriptor_set_layout])
                    .push_next(&mut count_allocate_info),
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

        let lighting_info = [vk::DescriptorBufferInfo::default()
            .buffer(lighting_buffer.handle())
            .range(vk::WHOLE_SIZE)];

        let lighting_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&lighting_info)
            .descriptor_count(1);

        let lights_info = [vk::DescriptorBufferInfo::default()
            .buffer(lights_buffer.handle())
            .range(vk::WHOLE_SIZE)];

        let lights_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&lights_info)
            .descriptor_count(1);

        unsafe {
            self.ctx.device.update_descriptor_sets(
                &[accel_write, texel_write, lighting_write, lights_write],
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
        lighting_device: &Buffer,
        lighting_readback: Option<&Buffer>,
    ) -> Result<()> {
        let sbt_buffer = self
            .shader_binding_table_buffer
            .as_ref()
            .context("SBT buffer not created")?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.ctx
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.ctx
                .device
                .begin_command_buffer(self.command_buffer, &begin_info)?;
        }

        let sbt_address = sbt_buffer.device_address();

        let sbt_raygen_region_direct = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + Self::SBT_GROUP_RAYGEN_DIRECT as u64 * sbt_handle_size)
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

        let gpu_span = info_span!("GPU", elapsed_ns = ::tracing::field::Empty);
        let entered = gpu_span.enter();
        unsafe {
            self.begin_timestamp_range(
                self.command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );

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

            let direct_span = info_span!("PASS: Direct Illumination");
            let _entered_direct = direct_span.enter();

            self.ray_tracing_pipeline_device.cmd_trace_rays(
                self.command_buffer,
                &sbt_raygen_region_direct,
                &sbt_miss_region,
                &sbt_hit_region,
                &sbt_call_region,
                texel_count as u32,
                1,
                1,
            );

            drop(_entered_direct);

            let gi_span = info_span!("PASS: Global Illumination");
            let _entered_gi = gi_span.enter();

            let bounce_count = 1;

            for _bounce in 0..bounce_count {
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(lighting_device.handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);

                self.ctx.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
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

            self.end_timestamp_range(
                self.command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );

            drop(_entered_gi);

            if let Some(readback) = lighting_readback {
                let to_transfer = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(lighting_device.handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                self.ctx.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[to_transfer],
                    &[],
                );

                self.ctx.device.cmd_copy_buffer(
                    self.command_buffer,
                    lighting_device.handle(),
                    readback.handle(),
                    &[vk::BufferCopy::default().size(readback.size())],
                );

                let to_host = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(readback.handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                self.ctx.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[to_host],
                    &[],
                );
            } else {
                let to_shader = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(lighting_device.handle())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                self.ctx.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[to_shader],
                    &[],
                );
            }

            self.ctx.device.end_command_buffer(self.command_buffer)?;
            self.ctx.device.queue_submit(
                self.ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.command_buffer])],
                vk::Fence::null(),
            )?;
            self.ctx.device.queue_wait_idle(self.ctx.queue)?;
        }

        gpu_span.record("elapsed_ns", self.read_elapsed_ns()?);
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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to BSP file
    bsp_path: PathBuf,

    /// Output BSP path
    #[arg(long, value_name = "PATH")]
    output: PathBuf,

    /// Use high dynamic range lumps
    #[arg(help_heading = "bsp", long)]
    hdr: bool,

    /// Enable kronos validation layer if available
    #[arg(help_heading = "vulkan", long)]
    validation_layer: bool,

    /// Force use of specific device id
    #[arg(help_heading = "vulkan", long)]
    device_id: Option<u32>,

    /// Lightmap resolution multiplier
    #[arg(help_heading = "bsp", long, default_value = "1")]
    lightmap_scale: u32,
}

fn luxel_to_world_matrix<'a>(face: &Face, bsp: &'a Bsp<'a>) -> Mat3 {
    let plane: Ref<'_, Plane> = face.associated(bsp);
    let tex: Ref<'_, TextureInfo> = face.associated(bsp);

    let s_vec = Vec3::from_array(tex.luxels[0].xyz);
    let t_vec = Vec3::from_array(tex.luxels[1].xyz);
    let normal = Vec3::from_array(plane.normal);

    let cross = t_vec.cross(s_vec);
    let det = -normal.dot(cross);

    if det.abs() < 1.0e-20 {
        warn!("face vectors parallel to face normal");
    }

    let inv_det = 1.0 / det;

    let s_axis = t_vec.cross(normal) * inv_det;
    let t_axis = normal.cross(s_vec) * inv_det;

    let [s_min, t_min] = face.lightmap.mins;

    let origin = cross * (-plane.dist * inv_det)
        - s_axis * (tex.luxels[0].offset - s_min as f32)
        - t_axis * (tex.luxels[1].offset - t_min as f32);

    Mat3::from_cols(s_axis, t_axis, origin)
}

#[instrument(skip_all, err)]
fn run(args: Args) -> Result<()> {
    let contents = std::fs::read(args.bsp_path)?;
    let bsp = Bsp::parse(&contents).map_err(|_| anyhow!("failed to parse BSP file"))?;

    let mut instance_layers = vec![];
    if args.validation_layer {
        instance_layers.push(c"VK_LAYER_KHRONOS_validation");
    }

    let device_layers = [
        acceleration_structure::NAME,
        deferred_host_operations::NAME,
        ray_tracing_pipeline::NAME,
        spirv_1_4::NAME,
        get_memory_requirements2::NAME,
        c"VK_KHR_ray_tracing_position_fetch",
    ];

    let ctx = Rc::new(VulkanContext::new(
        &instance_layers,
        &device_layers,
        args.device_id,
    )?);
    let mut app = Application::new(ctx.clone())?;

    let vertices = bsp.get_lump::<[Vertex]>(LumpDefinition::Vertices)?;
    let faces = bsp.get_lump::<[Face]>(LumpDefinition::Faces)?;

    let mut texels: Vec<TexelData> = Vec::new();

    let verts: &[[f32; 3]] = zerocopy::transmute_ref!(&*vertices);

    const NUDGE_DIST: f32 = 1.0;

    for face in faces.iter() {
        let plane: Ref<'_, Plane> = face.associated(&bsp);
        let normal = Vec3::from_array(plane.normal).normalize();

        let base_width = (face.lightmap.maxs[0] + 1) as u32;
        let base_height = (face.lightmap.maxs[1] + 1) as u32;
        let width = base_width * args.lightmap_scale;
        let height = base_height * args.lightmap_scale;

        let base_matrix = luxel_to_world_matrix(face, &bsp);
        let inv_scale = 1.0 / args.lightmap_scale as f32;
        let matrix = Mat3::from_cols(
            base_matrix.col(0) * inv_scale,
            base_matrix.col(1) * inv_scale,
            base_matrix.col(2),
        );

        let surface_edges: Ref<'_, [SurfaceEdge]> = face.associated(&bsp);
        let polygon: Vec<Vec3> = surface_edges
            .iter()
            .map(|se| {
                let idx = se.associated(&bsp).edge[usize::from(se.edge_index < 0)] as usize;
                Vec3::from_array(verts[idx])
            })
            .collect();

        let centroid = polygon.iter().copied().sum::<Vec3>() / polygon.len().max(1) as f32;

        let edges: Vec<(Vec3, Vec3)> = if polygon.len() >= 3 {
            (0..polygon.len())
                .map(|i| {
                    let a = polygon[i];
                    let b = polygon[(i + 1) % polygon.len()];
                    let edge_dir = (b - a).normalize();
                    let n = edge_dir.cross(normal);
                    let inward = if (centroid - a).dot(n) > 0.0 { n } else { -n };
                    (a, inward)
                })
                .collect()
        } else {
            Vec::new()
        };

        for t in 0..height {
            for s in 0..width {
                let mut world_pos = matrix * Vec3::new(s as f32, t as f32, 1f32);

                for &(edge_origin, inward) in &edges {
                    let dist = (world_pos - edge_origin).dot(inward);
                    if dist < NUDGE_DIST {
                        world_pos += inward * (NUDGE_DIST - dist);
                    }
                }

                texels.push(TexelData::new(world_pos, normal));
            }
        }
    }
    let mut texel_staging = Buffer::new(
        ctx.clone(),
        (size_of::<TexelData>() * texels.len()) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    texel_staging.store(&texels);

    let init_command_buffer = {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(ctx.pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        unsafe { ctx.device.allocate_command_buffers(&allocate_info) }?[0]
    };

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        ctx.device
            .begin_command_buffer(init_command_buffer, &begin_info)?;
    }

    let texel_buffer = Buffer::new(
        ctx.clone(),
        texel_staging.size(),
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    texel_buffer.cmd_copy_from(init_command_buffer, &texel_staging, texel_staging.size());

    let lighting_buffer_device = Buffer::new(
        ctx.clone(),
        (size_of::<AlignedVec3>() * texels.len()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let lighting_buffer_readback = Buffer::new(
        ctx.clone(),
        lighting_buffer_device.size(),
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // TODO: check for tools/toolsblocklight
    const INVISIBLE_FLAGS: u16 =
        SurfaceFlags::NODRAW | SurfaceFlags::TRIGGER | SurfaceFlags::HINT | SurfaceFlags::SKIP;

    const SKY_FLAGS: u16 = SurfaceFlags::SKY | SurfaceFlags::SKY2D;

    let face_category = |face: &Face| -> u8 {
        let tex: Ref<'_, TextureInfo> = face.associated(&bsp);
        let flags = tex.flags as u16;

        if flags & INVISIBLE_FLAGS != 0 {
            0 // invisible
        } else if flags & SKY_FLAGS != 0 {
            2 // sky face
        } else {
            1 // solid
        }
    };

    let triangulate_face = |face: &Face| -> Vec<u16> {
        let surface_edges: Ref<'_, [SurfaceEdge]> = face.associated(&bsp);
        let mut it = surface_edges.iter().map(|surface_edge| {
            surface_edge.associated(&bsp).edge[usize::from(surface_edge.edge_index < 0)]
        });
        let mut tris = Vec::new();
        let Some(pivot) = it.next() else { return tris };
        let Some(mut prev) = it.next() else {
            return tris;
        };
        for current in it {
            tris.extend([pivot, prev, current]);
            prev = current;
        }
        tris
    };

    let mut solid_indices: Vec<u16> = Vec::new();
    let mut sky_indices: Vec<u16> = Vec::new();

    for face in faces.iter() {
        match face_category(face) {
            1 => solid_indices.extend(triangulate_face(face)),
            2 => sky_indices.extend(triangulate_face(face)),
            _ => {}
        }
    }

    info!(
        "Geometry: {} solid triangles, {} sky triangles",
        solid_indices.len() / 3,
        sky_indices.len() / 3
    );

    let vertices: &[[f32; 3]] = zerocopy::transmute_ref!(&*vertices);
    let _setup_buffers = app.create_acceleration_structures(
        init_command_buffer,
        vertices,
        &solid_indices,
        &sky_indices,
    )?;

    let entity_string = {
        let entity_lump = bsp.lump(LumpDefinition::Entities);
        String::from_utf8_lossy(&entity_lump.data).into_owned()
    };
    let entities = valve_kv::parse(&entity_string).unwrap_or_default();

    let mut ambient_override: Option<Vec3> = None;
    for ent in &entities {
        if ent.properties.get("classname") == Some(&"light_environment") {
            if let Some(ambient_str) = ent.properties.get("_ambient") {
                let parts: Vec<f32> = ambient_str
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if parts.len() >= 4 {
                    let scale = parts[3] / 255.0;
                    ambient_override = Some(Vec3::new(
                        parts[0] * scale,
                        parts[1] * scale,
                        parts[2] * scale,
                    ));
                }
            }
            if let Some(spread) = ent.properties.get("SunSpreadAngle") {
                if let Ok(angle) = spread.parse::<f32>() {
                    let extent = (angle.to_radians()).sin();
                    info!("Sun angular extent from entity: {extent}");
                }
            }
            break;
        }
    }

    let mut lights: Vec<shared::Light> = Vec::new();

    let worldlights = bsp
        .get_lump::<[WorldLight]>(LumpDefinition::WorldLights)
        .context("Failed to parse worldlights lump")?;

    for (i, wl) in worldlights.iter().enumerate() {
        let Ok(ty) = EmitType::try_from(wl.ty) else {
            tracing::warn!(
                "Light {} is an unsupported type ({}) and was skipped.",
                i,
                wl.ty
            );
            continue;
        };

        let mut c = wl.constant_attn;
        let mut l = wl.linear_attn;
        let mut q = wl.quadratic_attn;

        let color = [wl.intensity[0], wl.intensity[1], wl.intensity[2], wl.radius];

        match ty {
            EmitType::Point | EmitType::Spotlight => {
                if c < 0.0001 && l < 0.0001 && q < 0.0001 {
                    c = 1.0;
                }
            }
            EmitType::Surface => {
                if c < 0.0001 && l < 0.0001 && q < 0.0001 {
                    q = 1.0;
                }
            }
            EmitType::SkyLight | EmitType::SkyAmbient => {
                c = 1.0;
                l = 0.0;
                q = 0.0;
            }
            EmitType::QuakeLight => {
                c = 0.0;
                l = 1.0;
                q = 0.0;
            }
        }

        let light = shared::Light {
            position: Vec3::new(wl.origin[0], wl.origin[1], wl.origin[2]).into(),
            color: Vec3::new(color[0], color[1], color[2]).into(),
            direction: Vec3::new(wl.normal[0], wl.normal[1], wl.normal[2]).into(),
            ty,
            radius: color[3],
            constant_attn: c,
            linear_attn: l,
            quadratic_attn: q,
            penumbra_start: wl.penumbra_start,
            penumbra_end: wl.penumbra_end,
            exponent: wl.exponent,
        };

        tracing::info!("Light {i}: {light:?}");

        lights.push(light);
    }

    if let Some(ambient_color) = ambient_override {
        for light in lights.iter_mut() {
            if light.ty == EmitType::SkyAmbient {
                light.color = ambient_color.into();
            }
        }
    }

    let mut lights_staging = Buffer::new(
        ctx.clone(),
        (size_of::<shared::Light>() * lights.len()) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    lights_staging.store(&lights);

    let lights_buffer_device = Buffer::new(
        ctx.clone(),
        lights_staging.size(),
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    lights_buffer_device.cmd_copy_from(init_command_buffer, &lights_staging, lights_staging.size());

    let init_to_rt_barrier = vk::MemoryBarrier::default()
        .src_access_mask(
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags::TRANSFER_WRITE,
        )
        .dst_access_mask(
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::SHADER_READ,
        );

    unsafe {
        ctx.device.cmd_pipeline_barrier(
            init_command_buffer,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::empty(),
            &[init_to_rt_barrier],
            &[],
            &[],
        );
        ctx.device.end_command_buffer(init_command_buffer)?;
        ctx.device.queue_submit(
            ctx.queue,
            &[vk::SubmitInfo::default().command_buffers(&[init_command_buffer])],
            vk::Fence::null(),
        )?;
        ctx.device.queue_wait_idle(ctx.queue)?;
        ctx.device
            .free_command_buffers(ctx.pool, &[init_command_buffer]);
    }

    let shader_group_count = app.create_pipeline()?;
    let sbt_handle_size = app.create_shader_binding_table(shader_group_count)?;

    app.create_descriptor_set(
        &texel_buffer,
        &lighting_buffer_device,
        &lights_buffer_device,
    )?;
    app.record_ray_tracing(
        sbt_handle_size,
        texels.len(),
        &lighting_buffer_device,
        Some(&lighting_buffer_readback),
    )?;

    let lighting: Vec<AlignedVec3> = lighting_buffer_readback.load(texels.len());
    let mut lighting_lump = bsp.lump_mut(LumpDefinition::Lighting);
    let mut lighting_hdr_lump = bsp.lump_mut(LumpDefinition::LightingHdr);

    // Drop immutable ref so we can take mutable ref
    drop(faces);
    let mut faces = bsp.get_lump_mut::<[Face]>(LumpDefinition::Faces)?;
    let mut faces_hdr = bsp.get_lump_mut::<[Face]>(LumpDefinition::FacesHdr).ok();

    if args.lightmap_scale > 1 {
        let scale = args.lightmap_scale as f32;
        let mut texinfos = bsp.get_lump_mut::<[TextureInfo]>(LumpDefinition::TextureInfo)?;
        for ti in texinfos.iter_mut() {
            for axis in &mut ti.luxels {
                axis.xyz[0] *= scale;
                axis.xyz[1] *= scale;
                axis.xyz[2] *= scale;
                axis.offset *= scale;
            }
        }

        for face in faces.iter_mut() {
            face.lightmap.mins[0] *= args.lightmap_scale as i32;
            face.lightmap.mins[1] *= args.lightmap_scale as i32;
        }
        if let Some(hdr) = &mut faces_hdr {
            for face in hdr.iter_mut() {
                face.lightmap.mins[0] *= args.lightmap_scale as i32;
                face.lightmap.mins[1] *= args.lightmap_scale as i32;
            }
        }
    }

    let encode_rgbexp32 = |color: Vec3| -> ColorRGBExp32 {
        let max_val = color.x.max(color.y).max(color.z);
        if max_val <= 0.0 {
            return ColorRGBExp32 {
                r: 0,
                g: 0,
                b: 0,
                exponent: 0,
            };
        }

        let mut exponent = max_val.log2().floor() as i32 + 1;
        exponent -= 8;

        let scalar = 2.0_f32.powi(-exponent);

        ColorRGBExp32 {
            r: (color.x * scalar).clamp(0.0, 255.0) as u8,
            g: (color.y * scalar).clamp(0.0, 255.0) as u8,
            b: (color.z * scalar).clamp(0.0, 255.0) as u8,
            exponent: exponent.clamp(-128, 127) as i8,
        }
    };

    let mut lightmap_samples: Vec<ColorRGBExp32> = Vec::with_capacity(lighting.len() + faces.len());
    let mut byte_offset: i32 = 0;
    let mut texel_offset = 0usize;

    for (i, face) in faces.iter_mut().enumerate() {
        let width = ((face.lightmap.maxs[0] + 1) as u32 * args.lightmap_scale) as usize;
        let height = ((face.lightmap.maxs[1] + 1) as u32 * args.lightmap_scale) as usize;
        let luxel_count = width * height;

        face.lightmap.maxs[0] = (width as i32) - 1;
        face.lightmap.maxs[1] = (height as i32) - 1;

        let end = (texel_offset + luxel_count).min(lighting.len());
        let face_texels = &lighting[texel_offset..end];

        let (mut sr, mut sg, mut sb) = (0.0_f32, 0.0_f32, 0.0_f32);
        for AlignedVec3(c) in face_texels {
            sr += c.x;
            sg += c.y;
            sb += c.z;
        }
        let n = luxel_count.max(1) as f32;
        let avg = Vec3::new(sr / n, sg / n, sb / n);

        lightmap_samples.push(encode_rgbexp32(avg));

        face.styles = [0, 255, 255, 255];
        face.light_offset = byte_offset + size_of::<ColorRGBExp32>() as i32;

        if let Some(hdr_faces) = &mut faces_hdr
            && let Some(hdr_face) = hdr_faces.get_mut(i)
        {
            hdr_face.styles = face.styles;
            hdr_face.light_offset = face.light_offset;
            hdr_face.lightmap.maxs = face.lightmap.maxs;
        }

        for AlignedVec3(color) in face_texels {
            lightmap_samples.push(encode_rgbexp32(*color));
        }

        byte_offset += (1 + luxel_count) as i32 * size_of::<ColorRGBExp32>() as i32;
        texel_offset += luxel_count;
    }

    let final_bytes = lightmap_samples.as_bytes().to_owned();
    lighting_lump.data = Cow::Owned(final_bytes.clone());
    lighting_hdr_lump.data = Cow::Owned(final_bytes);

    info!("Wrote {} bytes to lighting lump", lighting_lump.data.len());

    drop(faces);
    drop(faces_hdr);
    drop(lighting_lump);
    drop(lighting_hdr_lump);

    bsp.write_to_io(&mut File::create(&args.output)?)
        .context("writing to io failed")?;

    Ok(())
}

fn main() {
    let args = Args::parse();

    let timing_layer = TreeTimingLayer::default();

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("radiosity=info"))
        .add_directive("error".parse().unwrap());

    tracing_subscriber::registry()
        .with(timing_layer.clone())
        .with(env_filter)
        .init();

    std::panic::set_hook(Box::new(move |panic_info| {
        error!("{}", panic_info.payload_as_str().unwrap_or("unknown"))
    }));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(args)));
    timing_layer.print_tree(false);

    match result {
        Ok(Ok(())) => {}
        Ok(Err(_)) | Err(_) => std::process::exit(1),
    }
}
