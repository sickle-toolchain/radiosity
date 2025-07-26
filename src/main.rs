#![feature(gen_blocks)]
use std::{collections::HashSet, error::Error, ffi::CStr, os::raw::c_char, path::PathBuf};

use ash::{
    prelude::VkResult,
    util::Align,
    vk::{self, Packed24_8},
};
use bsp::Bsp;
use clap::Parser;
use lump_definitions::source::{Edge, Face, LumpDefinition, SurfaceEdge, Vertex};
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

    let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let command_pool = {
        let command_pool_create_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);

        unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .expect("Failed to create Command Pool!")
    };

    let acceleration_structure = ash::khr::acceleration_structure::Device::new(&instance, &device);

    // ...

    let vertices = bsp
        .lump_cast::<[Vertex], _>(LumpDefinition::Vertices)
        .expect("Failed to get vertices lump");

    let mut vertex_buffer = BufferResource::new(
        (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        &device,
        device_memory_properties,
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

    let mut index_buffer = BufferResource::new(
        (std::mem::size_of::<u16>() * indices.len()) as u64,
        vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        &device,
        device_memory_properties,
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

    let (blas, blas_buffer) = {
        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(indices.len() as u32 / 3);

        let geometries = [geometry];

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

        let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            acceleration_structure.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[build_range_info.primitive_count],
                &mut sizes_info,
            )
        };

        let blas_buffer = BufferResource::new(
            sizes_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &device,
            device_memory_properties,
        )?;

        let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(build_info.ty)
            .size(sizes_info.acceleration_structure_size)
            .buffer(blas_buffer.buffer)
            .offset(0);

        let blas =
            unsafe { acceleration_structure.create_acceleration_structure(&as_create_info, None) }
                .expect("Failed to create bottom-level acceleration structure");

        build_info.dst_acceleration_structure = blas;

        let scratch_buffer = BufferResource::new(
            sizes_info.acceleration_structure_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &device,
            device_memory_properties,
        )?;

        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_address(&device),
        };

        let build_command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(command_pool)
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

            acceleration_structure.cmd_build_acceleration_structures(
                build_command_buffer,
                &[build_info],
                &[&[build_range_info]],
            );
            device.end_command_buffer(build_command_buffer).unwrap();
            device
                .queue_submit(
                    graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[build_command_buffer])],
                    vk::Fence::null(),
                )
                .expect("queue submit failed");

            device.queue_wait_idle(graphics_queue).unwrap();
            device.free_command_buffers(command_pool, &[build_command_buffer]);
            scratch_buffer.destroy(&device);
        }
        (blas, blas_buffer)
    };

    let accel_handle = {
        let as_addr_info =
            vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blas);
        unsafe { acceleration_structure.get_acceleration_structure_device_address(&as_addr_info) }
    };

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
                device_handle: accel_handle,
            },
        }];
        let instance_buffer_size =
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * instances.len();

        let mut instance_buffer = BufferResource::new(
            instance_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &device,
            device_memory_properties,
        )?;

        instance_buffer.store(&instances, &device);

        instance_buffer
    };

    let (tlas, tlas_buffer) = {
        let build_range_info =
            vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1);

        let build_command_buffer = {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(command_pool)
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
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR);
            device.cmd_pipeline_barrier(
                build_command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }

        let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer.device_address(&device),
            });

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances });

        let geometries = [geometry];

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let mut sizes_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            acceleration_structure.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[build_range_info.primitive_count],
                &mut sizes_info,
            )
        };

        let tlas_buffer = BufferResource::new(
            sizes_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &device,
            device_memory_properties,
        )?;

        let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(build_info.ty)
            .size(sizes_info.acceleration_structure_size)
            .buffer(tlas_buffer.buffer)
            .offset(0);

        let tlas =
            unsafe { acceleration_structure.create_acceleration_structure(&as_create_info, None) }
                .expect("Failed to create top-level acceleration structure");

        build_info.dst_acceleration_structure = tlas;

        let scratch_buffer = BufferResource::new(
            sizes_info.build_scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &device,
            device_memory_properties,
        )?;

        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_address(&device),
        };

        unsafe {
            acceleration_structure.cmd_build_acceleration_structures(
                build_command_buffer,
                &[build_info],
                &[&[build_range_info]],
            );
            device.end_command_buffer(build_command_buffer).unwrap();
            device
                .queue_submit(
                    graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[build_command_buffer])],
                    vk::Fence::null(),
                )
                .expect("queue submit failed.");

            device.queue_wait_idle(graphics_queue).unwrap();
            device.free_command_buffers(command_pool, &[build_command_buffer]);
            scratch_buffer.destroy(&device);
        }

        (tlas, tlas_buffer)
    };

    unsafe {
        device.destroy_command_pool(command_pool, None);
    }

    unsafe {
        acceleration_structure.destroy_acceleration_structure(blas, None);
        acceleration_structure.destroy_acceleration_structure(tlas, None);
    }

    blas_buffer.destroy(&device);
    tlas_buffer.destroy(&device);

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

struct BufferResource {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl BufferResource {
    pub fn new(
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
        device: &ash::Device,
        device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    ) -> VkResult<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }?;

        let memory_req = unsafe { device.get_buffer_memory_requirements(buffer) };

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

        let memory = unsafe { device.allocate_memory(&allocate_info, None) }?;

        unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;

        Ok(Self { buffer, memory })
    }

    fn device_address(&self, device: &ash::Device) -> u64 {
        unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.buffer),
            )
        }
    }

    fn store<T: Copy>(&mut self, data: &[T], device: &ash::Device) {
        unsafe {
            let size = std::mem::size_of_val(data) as u64;
            let mapped_ptr = self.map(size, device);
            let mut mapped_slice = Align::new(mapped_ptr, std::mem::align_of::<T>() as u64, size);
            mapped_slice.copy_from_slice(data);
            self.unmap(device);
        }
    }

    fn map(&mut self, size: vk::DeviceSize, device: &ash::Device) -> *mut std::ffi::c_void {
        unsafe {
            let data: *mut std::ffi::c_void = device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            data
        }
    }

    fn unmap(&mut self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.memory);
        }
    }
    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}
