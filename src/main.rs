use std::{collections::HashSet, error::Error, ffi::CStr, os::raw::c_char, path::PathBuf};

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
            (instance.fp_v1_1().get_physical_device_features2)(physical_device, &mut features2);
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

    // ...

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
            let surface_edges = <Face as Associated<[SurfaceEdge]>>::associated(face, &bsp);
            let indices = surface_edges.iter().map(|surface_edge| {
                <SurfaceEdge as Associated<Edge>>::associated(surface_edge, &bsp).edge
                    [(surface_edge.edge_index < 0) as usize]
            });

            // Perform fan triangulation on list of vertices
            struct FanTriangulation<I, A>
            where
                I: Iterator<Item = A>,
            {
                iter: I,
                v0: Option<A>,
                prev: Option<A>,
            }

            impl<I, A> FanTriangulation<I, A>
            where
                I: Iterator<Item = A>,
            {
                fn new(mut iter: I) -> Self {
                    let v0 = iter.next();
                    let prev = iter.next();
                    FanTriangulation { iter, v0, prev }
                }
            }

            impl<I, A: Copy> Iterator for FanTriangulation<I, A>
            where
                I: Iterator<Item = A>,
            {
                type Item = [A; 3];

                fn next(&mut self) -> Option<Self::Item> {
                    if let Some(current) = self.iter.next() {
                        let triangle = [self.v0.unwrap(), self.prev.unwrap(), current];
                        self.prev = Some(current);
                        Some(triangle)
                    } else {
                        None
                    }
                }
            }

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

    unsafe {
        device.destroy_command_pool(command_pool, None);
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
