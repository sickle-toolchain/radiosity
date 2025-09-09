use std::{collections::HashSet, error::Error, ffi::CStr, os::raw::c_char, path::PathBuf, ptr};

use ash::{
    prelude::VkResult,
    vk::{self, Packed24_8},
};
use bsp::Bsp;
use clap::Parser;
use lump_definitions::source::{Edge, Face, LumpDefinition, SurfaceEdge, Vertex};

use radiosity::vulkan::{AccelerationStructure, ApplicationInfoExt, Buffer, VkContext};
use radiosity::Associated;

#[allow(dead_code)]
const SHADER: &[u8] = include_bytes!(env!("radiosity_shader.spv"));

#[derive(Parser, Debug)]
struct Args {
    /// Path to BSP file
    bsp_path: PathBuf,

    /// Use high dynamic range lumps
    #[arg(long)]
    hdr: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let contents = std::fs::read(args.bsp_path)?;
    let bsp = Bsp::parse(&contents).expect("failed to parse BSP file");

    let entry = unsafe { ash::Entry::load() }?;

    let instance = {
        let application_info = vk::ApplicationInfo::default()
            .application_from_env()
            .api_version(vk::API_VERSION_1_2);

        let instance_create_info =
            vk::InstanceCreateInfo::default().application_info(&application_info);

        unsafe { entry.create_instance(&instance_create_info, None) }
            .expect("failed to create instance!")
    };

    let physical_device = find_physical_device(
        &instance,
        &[
            ash::khr::acceleration_structure::NAME,
            ash::khr::deferred_host_operations::NAME,
            ash::khr::ray_tracing_pipeline::NAME,
        ],
    )?
    .expect("Failed to find physical device");

    let queue_family_index =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
            .into_iter()
            .enumerate()
            .find(|(_, device_properties)| {
                device_properties.queue_count > 0
                    && device_properties
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
            })
            .map(|(idx, _)| idx as u32)
            .expect("Failed to find queue family index");

    let device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let device: ash::Device = {
        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0])];

        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        unsafe {
            (instance.fp_v1_1().get_physical_device_features2)(physical_device, &raw mut features2);
        };

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .vulkan_memory_model(true);

        let mut as_feature = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);

        let mut raytracing_pipeline =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

        let enabled_extension_names = [
            ash::khr::acceleration_structure::NAME.as_ptr(),
            ash::khr::deferred_host_operations::NAME.as_ptr(),
            ash::khr::ray_tracing_pipeline::NAME.as_ptr(),
            vk::KHR_SPIRV_1_4_NAME.as_ptr(),
            vk::EXT_SCALAR_BLOCK_LAYOUT_NAME.as_ptr(),
            vk::KHR_GET_MEMORY_REQUIREMENTS2_NAME.as_ptr(),
        ];

        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut features2)
            .push_next(&mut features12)
            .push_next(&mut as_feature)
            .push_next(&mut raytracing_pipeline)
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&enabled_extension_names);

        unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .expect("Failed to create logical Device!")
    };

    let device_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let command_pool = {
        let command_pool_create_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);

        unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .expect("Failed to create Command Pool!")
    };

    let vk_ctx = VkContext::new(
        &device,
        &device_queue,
        &command_pool,
        &device_memory_properties,
    )?;

    let as_device = ash::khr::acceleration_structure::Device::new(&instance, &device);
    let rt_pipeline_device = ash::khr::ray_tracing_pipeline::Device::new(&instance, &device);
    let mut rt_pipeline_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    {
        let mut physical_device_properties2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut rt_pipeline_properties);

        unsafe {
            instance
                .get_physical_device_properties2(physical_device, &mut physical_device_properties2);
        }
    }

    let vertices = bsp
        .lump_cast::<[Vertex], _>(LumpDefinition::Vertices)
        .expect("Failed to get vertices lump");

    let mut vertex_buffer = Buffer::new(
        &vk_ctx,
        (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    vertex_buffer.store(&vertices, &device);

    let faces = bsp
        .lump_cast::<[Face], _>(LumpDefinition::Faces)
        .expect("Failed to get faces lump");

    let indices = faces
        .iter()
        .flat_map(|face| {
            struct FanTriangulation<I, A>
            where
                I: Iterator<Item = A>,
            {
                iter: I,
                pivot: Option<A>,
                prev: Option<A>,
            }

            impl<I, A> FanTriangulation<I, A>
            where
                I: Iterator<Item = A>,
            {
                fn new(mut iter: I) -> Self {
                    let pivot = iter.next();
                    let prev = iter.next();
                    FanTriangulation { iter, pivot, prev }
                }
            }

            impl<I, A: Copy> Iterator for FanTriangulation<I, A>
            where
                I: Iterator<Item = A>,
            {
                type Item = [A; 3];

                fn next(&mut self) -> Option<Self::Item> {
                    if let (Some(pivot), Some(prev), Some(current)) =
                        (self.pivot, self.prev, self.iter.next())
                    {
                        let triangle = [pivot, prev, current];
                        self.prev = Some(current);
                        Some(triangle)
                    } else {
                        None
                    }
                }
            }

            let surface_edges = <Face as Associated<[SurfaceEdge]>>::associated(face, &bsp);
            let indices = surface_edges.iter().map(|surface_edge| {
                <SurfaceEdge as Associated<Edge>>::associated(surface_edge, &bsp).edge
                    [usize::from(surface_edge.edge_index < 0)]
            });

            FanTriangulation::new(indices).collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    let mut index_buffer = Buffer::new(
        &vk_ctx,
        (std::mem::size_of::<u16>() * indices.len()) as u64,
        vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    index_buffer.store(&indices, &device);

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: vertex_buffer.device_address(&device),
                })
                .max_vertex(vertices.len() as u32 - 1)
                .vertex_stride(std::mem::size_of::<[f32; 3]>() as u64)
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: index_buffer.device_address(&device),
                })
                .index_type(vk::IndexType::UINT16),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let blas = AccelerationStructure::new(
        &vk_ctx,
        &as_device,
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        &[geometry],
        &[vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(indices.len() as u32 / 3)],
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
    )?;

    let instance_buffer = {
        let instances = [vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            instance_custom_index_and_mask: Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas.device_address(&as_device),
            },
        }];
        let instance_buffer_size =
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * instances.len();

        let mut instance_buffer = Buffer::new(
            &vk_ctx,
            instance_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        instance_buffer.store(&instances, &device);

        instance_buffer
    };

    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: instance_buffer.device_address(&device),
        });

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances });

    let tlas = AccelerationStructure::new(
        &vk_ctx,
        &as_device,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        &[geometry],
        &[vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1)],
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
    )?;

    // ...

    let (descriptor_set_layout, graphics_pipeline, pipeline_layout, shader_group_count) = {
        let binding_flags_inner = [vk::DescriptorBindingFlagsEXT::empty()];

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
            .binding_flags(&binding_flags_inner);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&[vk::DescriptorSetLayoutBinding::default()
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                        .binding(0)])
                    .push_next(&mut binding_flags),
                None,
            )
        }?;

        let shader_module_create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: SHADER.len(),
            p_code: SHADER.as_ptr().cast::<u32>(),
            ..Default::default()
        };

        let shader_module =
            unsafe { device.create_shader_module(&shader_module_create_info, None) }?;

        let layouts = vec![descriptor_set_layout];
        let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&layout_create_info, None) }.unwrap();

        let shader_groups = vec![
            // group0 = [ raygen ]
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // group1 = [ chit ]
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(1)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // group2 = [ miss ]
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
            rt_pipeline_device.create_ray_tracing_pipelines(
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
            device.destroy_shader_module(shader_module, None);
        }

        (
            descriptor_set_layout,
            pipeline,
            pipeline_layout,
            shader_groups.len(),
        )
    };

    let command_buffer = {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }?[0]
    };

    {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }?;
    }

    let handle_size_aligned = u64::from(
        (rt_pipeline_properties.shader_group_handle_size
            + rt_pipeline_properties.shader_group_base_alignment
            - 1)
            & !(rt_pipeline_properties.shader_group_base_alignment - 1),
    );

    let shader_binding_table_buffer = {
        let incoming_table_data = unsafe {
            rt_pipeline_device.get_ray_tracing_shader_group_handles(
                graphics_pipeline,
                0,
                shader_group_count as u32,
                shader_group_count * rt_pipeline_properties.shader_group_handle_size as usize,
            )
        }
        .unwrap();

        let table_size = shader_group_count * handle_size_aligned as usize;
        let mut table_data = vec![0u8; table_size];

        for i in 0..shader_group_count {
            table_data[i * handle_size_aligned as usize
                ..i * handle_size_aligned as usize
                    + rt_pipeline_properties.shader_group_handle_size as usize]
                .copy_from_slice(
                    &incoming_table_data[i * rt_pipeline_properties.shader_group_handle_size
                        as usize
                        ..i * rt_pipeline_properties.shader_group_handle_size as usize
                            + rt_pipeline_properties.shader_group_handle_size as usize],
                );
        }

        let mut shader_binding_table_buffer = Buffer::new(
            &vk_ctx,
            table_size as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        shader_binding_table_buffer.store(&table_data, &device);

        shader_binding_table_buffer
    };

    let descriptor_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        descriptor_count: 1,
    }];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&descriptor_sizes)
        .max_sets(1);

    let descriptor_pool =
        unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }.unwrap();

    let mut count_allocate_info =
        vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(&[1]);

    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_set_layout])
                .push_next(&mut count_allocate_info),
        )
    }?[0];

    let accel_structs = [tlas.handle()];
    let mut accel_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
        .acceleration_structures(&accel_structs);

    let mut accel_write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .push_next(&mut accel_info);

    accel_write.descriptor_count = 1;

    unsafe {
        device.update_descriptor_sets(&[accel_write], &[]);
    }

    {
        // |[ raygen shader ]|[ hit shader  ]|[ miss shader ]|
        // |                 |               |               |
        // | 0               | 1             | 2             | 3

        let sbt_address = shader_binding_table_buffer.device_address(&device);

        let sbt_raygen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address)
            .size(handle_size_aligned)
            .stride(handle_size_aligned);

        let sbt_miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + 2 * handle_size_aligned)
            .size(handle_size_aligned)
            .stride(handle_size_aligned);

        let sbt_hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + handle_size_aligned)
            .size(handle_size_aligned)
            .stride(handle_size_aligned);

        let sbt_call_region = vk::StridedDeviceAddressRegionKHR::default();

        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                graphics_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            rt_pipeline_device.cmd_trace_rays(
                command_buffer,
                &sbt_raygen_region,
                &sbt_miss_region,
                &sbt_hit_region,
                &sbt_call_region,
                1920,
                1080,
                1,
            );
            device.end_command_buffer(command_buffer).unwrap();
        }
    }

    {
        let command_buffers = [command_buffer];
        let submit_infos = [vk::SubmitInfo::default().command_buffers(&command_buffers)];

        unsafe {
            device
                .queue_submit(device_queue, &submit_infos, vk::Fence::null())
                .expect("Failed to execute queue submit.");

            device.queue_wait_idle(device_queue).unwrap();
        }
    };

    unsafe {
        device.destroy_command_pool(command_pool, None);
    }

    unsafe {
        device.destroy_descriptor_pool(descriptor_pool, None);
        shader_binding_table_buffer.destroy(&device);
        device.destroy_pipeline(graphics_pipeline, None);
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
    }

    unsafe {
        device.destroy_pipeline_layout(pipeline_layout, None);
    }

    blas.destroy(&as_device, &device);
    tlas.destroy(&as_device, &device);

    instance_buffer.destroy(&device);
    vertex_buffer.destroy(&device);
    index_buffer.destroy(&device);

    unsafe {
        device.destroy_device(None);
    }

    unsafe {
        instance.destroy_instance(None);
    }

    Ok(())
}

fn find_physical_device(
    instance: &ash::Instance,
    required_extensions: &[&CStr],
) -> VkResult<Option<vk::PhysicalDevice>> {
    let device = unsafe { instance.enumerate_physical_devices() }?
        .into_iter()
        .find(|&physical_device| {
            unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .map(|exts| {
                    let set: HashSet<&CStr> = exts
                        .iter()
                        .map(|ext| unsafe { CStr::from_ptr(&ext.extension_name as *const c_char) })
                        .collect();

                    required_extensions.iter().all(|ext| set.contains(ext))
                })
                .unwrap_or(false)
        });

    Ok(device)
}
