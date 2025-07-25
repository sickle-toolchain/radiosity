#![feature(gen_blocks)]
use std::{collections::HashSet, error::Error, ffi::CStr, os::raw::c_char, path::PathBuf};

use ash::{prelude::VkResult, util::Align, vk};
use bsp::Bsp;
use clap::Parser;
use lump_definitions::source::{LumpDefinition, Vertex};

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

trait ApplicationInfoExt {
    fn application_from_env(self) -> Self;
}

impl<'a> ApplicationInfoExt for vk::ApplicationInfo<'a> {
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

trait PhysicalDeviceMemoryPropertiesExt {
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

    // ...

    let vertices = bsp
        .lump_cast::<[Vertex], _>(LumpDefinition::Vertices)
        .expect("Failed to get vertices lump");

    let vertex_buffer = {
        let buffer_size = (std::mem::size_of::<Vertex>() * vertices.len()) as u64;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None) }
            .expect("Failed to create vertex buffer");

        let memory_req = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mut memory_allocate_flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_req.size)
            .memory_type_index(
                device_memory_properties
                    .mem_ty_idx(
                        memory_req.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                    .expect("Failed to get required memory type"),
            )
            .push_next(&mut memory_allocate_flags_info);

        let memory = unsafe { device.allocate_memory(&allocate_info, None) }
            .expect("Failed to allocate vertex buffer on device");

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .expect("Failed to bind vertex buffer to device");

        // Write vertices to vertex_buffer
        unsafe {
            let mapped = device.map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())?;
            let mut slice = Align::new(mapped, std::mem::align_of::<Vertex>() as u64, buffer_size);
            slice.copy_from_slice(&vertices);
            device.unmap_memory(memory);
        }

        buffer
    };

    // 1. Iterate faces
    // 2. Get edge vertices
    // 3. Perform fan triangulation
    let index_buffer = {};

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: unsafe {
                        device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer),
                        )
                    },
                })
                .max_vertex(vertices.len() as u32 - 1)
                .vertex_stride(std::mem::size_of::<[f32; 3]>() as u64)
                .vertex_format(vk::Format::R32G32B32_SFLOAT), // .index_data(vk::DeviceOrHostAddressConstKHR {
                                                              //     device_address: unsafe {
                                                              //         device.get_buffer_device_address(
                                                              //             &vk::BufferDeviceAddressInfo::default().buffer(index_buffer),
                                                              //         )
                                                              //     },
                                                              // })
                                                              // .index_type(vk::IndexType::UINT32),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

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
