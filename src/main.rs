use std::borrow::Cow;
use std::ffi::CStr;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::ptr;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use tracing::{info, info_span, instrument, warn};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_timing::{AlternateScreenGuard, TreeTimingLayer};
use zerocopy::IntoBytes;

use ash::ext::scalar_block_layout;
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
use spirv_std::glam::{Mat3, Mat4, Vec3};

use bsp::Bsp;
use lump_definitions::source::{
    ColorRGBExp32, Edge, Face, LumpDefinition, Plane, SurfaceEdge, TextureData, TextureInfo, Vertex,
};

use radiosity::Associated;
use radiosity::vulkan::{Buffer, GeometryIndex, VulkanContext};
use shared::{AlignedVec3, TexelData};

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
    pub tlas: AccelerationStructureKHR,
    pub tlas_buffer: Option<Buffer>,

    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub shader_binding_table_buffer: Option<Buffer>,

    pub command_buffer: vk::CommandBuffer,

    pub timestamp_query_pool: QueryPool,
}

impl Application<'_> {
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
            tlas: AccelerationStructureKHR::null(),
            tlas_buffer: None,
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

    // Create triangular acceleration structure geometry
    #[instrument(skip_all)]
    fn create_triangle_geometry<I>(
        &'_ self,
        vertices: &[Vertex],
        indices: &[I],
    ) -> Result<(AccelerationStructureGeometryKHR<'_>, Buffer, Buffer)>
    where
        I: GeometryIndex + Copy,
    {
        const INPUT_BUFFER_FLAGS: BufferUsageFlags = BufferUsageFlags::from_raw(
            BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw()
                | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR.as_raw(),
        );
        const MEMORY_PROPERTIES: MemoryPropertyFlags = MemoryPropertyFlags::from_raw(
            MemoryPropertyFlags::HOST_VISIBLE.as_raw()
                | MemoryPropertyFlags::HOST_COHERENT.as_raw(),
        );

        let mut vertex_buffer = Buffer::new(
            self.ctx.clone(),
            size_of_val(vertices) as vk::DeviceSize,
            INPUT_BUFFER_FLAGS | BufferUsageFlags::VERTEX_BUFFER,
            MEMORY_PROPERTIES,
        )?;
        vertex_buffer.store(vertices);

        let mut index_buffer = Buffer::new(
            self.ctx.clone(),
            size_of_val(indices) as vk::DeviceSize,
            INPUT_BUFFER_FLAGS | BufferUsageFlags::INDEX_BUFFER,
            MEMORY_PROPERTIES,
        )?;
        index_buffer.store(indices);

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: vertex_buffer.device_address(),
                    })
                    .max_vertex(vertices.len() as u32 - 1)
                    .vertex_stride(size_of::<[f32; 3]>() as u64)
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: index_buffer.device_address(),
                    })
                    .index_type(I::vk_index_type()),
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        Ok((geometry, vertex_buffer, index_buffer))
    }

    // Create instance acceleration structure geometry
    #[instrument(skip_all)]
    fn create_instance_geometry(
        &'_ self,
        instance: AccelerationStructureKHR,
        matrix: [f32; 12],
    ) -> Result<(AccelerationStructureGeometryKHR<'_>, Buffer)> {
        const INPUT_BUFFER_FLAGS: BufferUsageFlags = BufferUsageFlags::from_raw(
            BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw()
                | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR.as_raw(),
        );
        const MEMORY_PROPERTIES: MemoryPropertyFlags = MemoryPropertyFlags::from_raw(
            MemoryPropertyFlags::HOST_VISIBLE.as_raw()
                | MemoryPropertyFlags::HOST_COHERENT.as_raw(),
        );

        let device_handle = unsafe {
            self.acceleration_structure_device
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(instance),
                )
        };

        let instances = [vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR { matrix },
            instance_custom_index_and_mask: Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle,
            },
        }];

        let mut instance_buffer = Buffer::new(
            self.ctx.clone(),
            size_of_val(&instances) as vk::DeviceSize,
            INPUT_BUFFER_FLAGS,
            MEMORY_PROPERTIES,
        )?;
        instance_buffer.store(&instances);

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
        structure_type: AccelerationStructureTypeKHR,
        geometries: &[AccelerationStructureGeometryKHR],
        build_range_info: &[AccelerationStructureBuildRangeInfoKHR],
        flags: BuildAccelerationStructureFlagsKHR,
    ) -> Result<(AccelerationStructureKHR, Buffer)> {
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

        let build_command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(self.ctx.pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers =
                unsafe { self.ctx.device.allocate_command_buffers(&allocate_info) }?;
            command_buffers[0]
        };

        let gpu_span = info_span!("GPU", elapsed_ns = ::tracing::field::Empty);
        let entered = gpu_span.enter();

        unsafe {
            self.ctx.device.begin_command_buffer(
                build_command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            self.begin_timestamp_range(
                build_command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            );

            self.acceleration_structure_device
                .cmd_build_acceleration_structures(
                    build_command_buffer,
                    &[build_info],
                    &[build_range_info],
                );

            self.end_timestamp_range(
                build_command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            );
            self.ctx.device.end_command_buffer(build_command_buffer)?;
            self.ctx.device.queue_submit(
                self.ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&[build_command_buffer])],
                vk::Fence::null(),
            )?;

            self.ctx.device.queue_wait_idle(self.ctx.queue)?;
            self.ctx
                .device
                .free_command_buffers(self.ctx.pool, &[build_command_buffer]);
        }
        gpu_span.record("elapsed_ns", self.read_elapsed_ns()?);
        drop(entered);

        Ok((acceleration_structure, buffer))
    }

    #[instrument(skip_all)]
    pub fn create_acceleration_structures<I>(
        &mut self,
        vertices: &[Vertex],
        indices: &[I],
    ) -> Result<()>
    where
        I: GeometryIndex + Copy,
    {
        let (blas_geometry, _vertex_buffer, _index_buffer) =
            self.create_triangle_geometry(vertices, indices)?;
        let (blas, blas_buffer) = self.create_as(
            AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &[blas_geometry],
            &[vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(indices.len() as u32 / 3)],
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        )?;

        let identity_matrix: [f32; 12] = std::array::from_fn(|i| Mat4::IDENTITY.to_cols_array()[i]);
        let (tlas_geometry, _tlas_geometry_buffer) =
            self.create_instance_geometry(blas, identity_matrix)?;

        let (tlas, tlas_buffer) = self.create_as(
            AccelerationStructureTypeKHR::TOP_LEVEL,
            &[tlas_geometry],
            &[vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1)],
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        )?;

        self.blas = blas;
        self.blas_buffer = Some(blas_buffer);
        self.tlas = tlas;
        self.tlas_buffer = Some(tlas_buffer);
        Ok(())
    }

    #[instrument(skip_all)]
    pub fn create_pipeline(&mut self) -> Result<usize> {
        let binding_flags_inner = [
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
        ];

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
            .binding_flags(&binding_flags_inner);

        let descriptor_set_layout = unsafe {
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
                    ])
                    .push_next(&mut binding_flags),
                None,
            )
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

        let layouts = vec![descriptor_set_layout];
        let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

        let pipeline_layout = unsafe {
            self.ctx
                .device
                .create_pipeline_layout(&layout_create_info, None)
        }?;

        let shader_groups = vec![
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(1)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(2)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        let shader_stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(shader_module)
                .name(c"ray_generation"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(shader_module)
                .name(c"closest_hit"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(shader_module)
                .name(c"miss"),
        ];

        let pipeline = unsafe {
            self.ray_tracing_pipeline_device
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[vk::RayTracingPipelineCreateInfoKHR::default()
                        .stages(&shader_stages)
                        .groups(&shader_groups)
                        .max_pipeline_ray_recursion_depth(1)
                        .layout(pipeline_layout)],
                    None,
                )
        }
        .map_err(|(_, result)| result)?[0];

        unsafe {
            self.ctx.device.destroy_shader_module(shader_module, None);
        }

        self.descriptor_set_layout = descriptor_set_layout;
        self.pipeline_layout = pipeline_layout;
        self.pipeline = pipeline;

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
    ) -> Result<()> {
        let descriptor_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
        }];

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

        unsafe {
            self.ctx
                .device
                .update_descriptor_sets(&[accel_write, texel_write, lighting_write], &[]);
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
        debug_assert!(self.pipeline != vk::Pipeline::null());
        debug_assert!(self.pipeline_layout != vk::PipelineLayout::null());
        debug_assert!(self.descriptor_set != vk::DescriptorSet::null());

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
        let sbt_raygen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address)
            .size(sbt_handle_size)
            .stride(sbt_handle_size);
        let sbt_hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + sbt_handle_size)
            .size(sbt_handle_size)
            .stride(sbt_handle_size);
        let sbt_miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + 2 * sbt_handle_size)
            .size(sbt_handle_size)
            .stride(sbt_handle_size);
        let sbt_call_region = vk::StridedDeviceAddressRegionKHR::default();

        let gpu_span = info_span!("GPU", elapsed_ns = ::tracing::field::Empty);
        let entered = gpu_span.enter();
        unsafe {
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

            self.begin_timestamp_range(
                self.command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );

            self.ray_tracing_pipeline_device.cmd_trace_rays(
                self.command_buffer,
                &sbt_raygen_region,
                &sbt_miss_region,
                &sbt_hit_region,
                &sbt_call_region,
                texel_count as u32,
                1,
                1,
            );

            self.end_timestamp_range(
                self.command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );

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

    /// Dump BLAS geometry to OBJ file
    #[arg(help_heading = "vulkan", long, value_name = "PATH")]
    dump_blas_geometry: Option<PathBuf>,
}

fn luxel_to_world_matrix<'a>(face: &Face, bsp: &'a Bsp<'a>) -> Mat3 {
    let plane = <Face as Associated<Plane>>::associated(face, bsp);
    let tex = <Face as Associated<TextureInfo>>::associated(face, bsp);

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

#[instrument(err)]
fn run() -> Result<()> {
    let args = Args::parse();

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
        scalar_block_layout::NAME,
        get_memory_requirements2::NAME,
    ];

    let ctx = Rc::new(VulkanContext::new(
        &instance_layers,
        &device_layers,
        args.device_id,
    )?);
    let mut app = Application::new(ctx.clone())?;

    let vertices = bsp
        .lump_cast::<[Vertex], _>(LumpDefinition::Vertices)
        .map_err(|_| anyhow!("Failed to get vertices lump"))?;

    let faces = bsp
        .lump_cast::<[Face], _>(LumpDefinition::Faces)
        .map_err(|_| anyhow!("Failed to get faces lump"))?;

    let mut texels: Vec<TexelData> = Vec::new();

    for face in faces.iter() {
        let plane = <Face as Associated<Plane>>::associated(face, &bsp);
        let normal = Vec3::from_array(plane.normal).normalize();

        let width = (face.lightmap.maxs[0] + 1) as u32;
        let height = (face.lightmap.maxs[1] + 1) as u32;

        let matrix = luxel_to_world_matrix(face, &bsp);
        for t in 0..height {
            for s in 0..width {
                let world_pos = matrix * Vec3::new(s as f32, t as f32, 1f32);
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

    let texel_buffer = Buffer::new(
        ctx.clone(),
        texel_staging.size(),
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    texel_buffer.copy_from(&texel_staging, texel_staging.size())?;

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

    let indices = {
        let texture_info_lump = bsp
            .lump_cast::<[TextureInfo], _>(LumpDefinition::TextureInfo)
            .map_err(|_| anyhow!("Failed to get texture info lump"))?;
        let texture_data_lump = bsp
            .lump_cast::<[TextureData], _>(LumpDefinition::TextureData)
            .map_err(|_| anyhow!("Failed to get texture data lump"))?;
        let texture_data_string_table = bsp
            .lump_cast::<[u32], _>(LumpDefinition::TextureDataStringTable)
            .map_err(|_| anyhow!("Failed to get texture data string table lump"))?;
        let texture_data_string_data = bsp
            .lump_cast::<[u8], _>(LumpDefinition::TextureDataStringData)
            .map_err(|_| anyhow!("Failed to get texture data string data lump"))?;

        let capacity = faces
            .iter()
            .map(|face| {
                let surface_edges = <Face as Associated<[SurfaceEdge]>>::associated(face, &bsp);
                // n edges produces (n - 2) triangles, each triangle has 3 indices
                let triangle_count = surface_edges.len().saturating_sub(2);
                triangle_count * 3
            })
            .sum();

        let mut indices = Vec::with_capacity(capacity);
        for face in faces.iter() {
            let texture_data_idx =
                texture_info_lump[face.texture_info_index as usize].texture_data_index;
            let table_idx = texture_data_lump[texture_data_idx as usize].name_index;
            let data_idx = texture_data_string_table[table_idx as usize];
            let texture_name =
                CStr::from_bytes_until_nul(&texture_data_string_data[data_idx as usize..])?
                    .to_string_lossy();
            if texture_name.eq_ignore_ascii_case("TOOLS/TOOLSSKYBOX") {
                continue;
            }

            let surface_edges = <Face as Associated<[SurfaceEdge]>>::associated(face, &bsp);

            let mut it = surface_edges.iter().map(|surface_edge| {
                <SurfaceEdge as Associated<Edge>>::associated(surface_edge, &bsp).edge
                    [usize::from(surface_edge.edge_index < 0)]
            });

            let Some(pivot) = it.next() else { continue };
            let Some(mut prev) = it.next() else { continue };

            for current in it {
                indices.extend([pivot, prev, current]);
                prev = current;
            }
        }
        indices
    };

    if let Some(obj_path) = &args.dump_blas_geometry {
        let mut obj_file = File::create(obj_path).context("Failed to create OBJ file")?;

        for vertex in vertices.iter() {
            writeln!(
                obj_file,
                "v {} {} {}",
                vertex.point[0], vertex.point[1], vertex.point[2]
            )?;
        }

        writeln!(obj_file)?;

        for triangle in indices.chunks(3) {
            writeln!(
                obj_file,
                "f {} {} {}",
                triangle[0] + 1,
                triangle[1] + 1,
                triangle[2] + 1
            )?;
        }

        info!("Dumped BLAS geometry to {}", obj_path.display());
    }

    app.create_acceleration_structures(&vertices, &indices)?;

    let shader_group_count = app.create_pipeline()?;
    let sbt_handle_size = app.create_shader_binding_table(shader_group_count)?;

    app.create_descriptor_set(&texel_buffer, &lighting_buffer_device)?;
    app.record_ray_tracing(
        sbt_handle_size,
        texels.len(),
        &lighting_buffer_device,
        Some(&lighting_buffer_readback),
    )?;

    let lighting: Vec<AlignedVec3> = lighting_buffer_readback.load(texels.len());
    let mut lighting_lump = bsp.lump_mut(LumpDefinition::Lighting);

    // Drop immutable ref so we can take mutable ref
    drop(faces);
    let mut faces = bsp
        .lump_cast_mut::<[Face], _>(LumpDefinition::Faces)
        .map_err(|_| anyhow!("Failed to get faces lump"))?;

    let lightmap: Vec<_> = lighting
        .into_iter()
        .map(|AlignedVec3(color)| ColorRGBExp32 {
            r: (color.x * 255.0) as u8,
            g: (color.y * 255.0) as u8,
            b: (color.z * 255.0) as u8,
            exponent: 0,
        })
        .collect();

    faces.iter_mut().fold(0usize, |offset, face| {
        let width = (face.lightmap.maxs[0] + 1) as usize;
        let height = (face.lightmap.maxs[1] + 1) as usize;
        let luxels = width * height;

        // enable style 0
        face.styles = [0, 255, 255, 255];
        face.light_offset = (offset * size_of::<ColorRGBExp32>()) as i32;

        offset + luxels
    });

    lighting_lump.data = Cow::Owned(lightmap.as_bytes().to_owned());
    info!("Wrote {} bytes to lighting lump", lighting_lump.data.len());

    drop(faces);
    drop(lighting_lump);

    bsp.write_to_io(&mut File::create(&args.output)?)
        .context("writing to io failed")?;

    Ok(())
}

fn main() {
    let timing_layer = TreeTimingLayer::default();

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("radiosity=info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(timing_layer.clone())
        .init();

    let _screen_guard = AlternateScreenGuard::new(move || timing_layer.print_tree(false));
    if let Err(_error) = run() {
        std::process::exit(1);
    }
}
