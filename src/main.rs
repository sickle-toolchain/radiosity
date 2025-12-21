use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::{CStr, c_void};
use std::fs::File;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use log::{Level, LevelFilter, error, info, log, warn};
use zerocopy::IntoBytes;

use ash::ext::{debug_utils, scalar_block_layout};
use ash::khr::{
    acceleration_structure, deferred_host_operations, get_memory_requirements2,
    ray_tracing_pipeline, spirv_1_4,
};
use ash::prelude::VkResult;
use ash::vk::{self, Packed24_8, PhysicalDevice};
use ash::{Device, Entry, Instance};
use spirv_std::glam::Vec3;

use bsp::Bsp;
use lump_definitions::source::{
    ColorRGBExp32, Edge, Face, LumpDefinition, Plane, SurfaceEdge, TextureData, TextureInfo, Vertex,
};

use radiosity::Associated;
use radiosity::vulkan::{AccelerationStructure, ApplicationInfoExt, Buffer, VkContext};
use shared::{AlignedVec3, TexelData};

const SHADER: &[u8] = include_bytes!(env!("radiosity_shader.spv"));

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to BSP file
    bsp_path: PathBuf,

    /// Use high dynamic range lumps
    #[arg(help_heading = "bsp", long)]
    hdr: bool,

    /// Enable kronos validation layer if available
    #[arg(help_heading = "vulkan", long)]
    validation_layer: bool,

    /// Force use of specific device id
    #[arg(help_heading = "vulkan", long)]
    device_id: Option<u32>,
}

pub unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let level = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Level::Debug,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::Error,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Level::Info,
        _ => unreachable!(),
    };

    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) }.to_string_lossy();
    log!(target: "vulkan", level, "{message}");

    vk::FALSE
}

fn create_vk_instance(entry: &Entry, layers: Vec<&CStr>) -> Result<Instance> {
    let instance_layer_properties = unsafe { entry.enumerate_instance_layer_properties() }?;
    let instance_layers: Vec<&CStr> = instance_layer_properties
        .iter()
        .filter_map(|p| p.layer_name_as_c_str().ok())
        .collect();

    for layer in &layers {
        if !instance_layers.contains(layer) {
            bail!("Layer '{}' is not suppported", layer.to_string_lossy());
        }
    }

    let instance = {
        let application_info = vk::ApplicationInfo::default()
            .application_from_env()
            .api_version(vk::API_VERSION_1_2);

        let enabled_extension_names = vec![debug_utils::NAME.as_ptr()];
        let mut debug_utils_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        let enabled_layer_names = layers.iter().map(|l| l.as_ptr()).collect::<Vec<_>>();

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(enabled_layer_names.as_slice())
            .enabled_extension_names(enabled_extension_names.as_slice())
            .push_next(&mut debug_utils_create_info);

        unsafe { entry.create_instance(&instance_create_info, None) }
            .context("Failed to create instance")?
    };

    Ok(instance)
}

pub trait PhysicalDeviceExt {
    fn find_queue_family_idx(
        &self,
        instance: &Instance,
        pred: fn(&vk::QueueFamilyProperties) -> bool,
    ) -> Option<u32>;
}

impl PhysicalDeviceExt for PhysicalDevice {
    fn find_queue_family_idx(
        &self,
        instance: &Instance,
        pred: fn(&vk::QueueFamilyProperties) -> bool,
    ) -> Option<u32> {
        unsafe { instance.get_physical_device_queue_family_properties(*self) }
            .into_iter()
            .enumerate()
            .find(|(_, device_properties)| pred(device_properties))
            .map(|(idx, _)| idx as u32)
    }
}

pub trait InstanceExt {
    fn physical_device_by_id(&self, id: u32) -> VkResult<Option<PhysicalDevice>>;
    fn find_physical_device(
        &self,
        required_extensions: &[&CStr],
    ) -> VkResult<Option<PhysicalDevice>>;
}

impl InstanceExt for Instance {
    fn physical_device_by_id(&self, id: u32) -> VkResult<Option<PhysicalDevice>> {
        let device = unsafe { self.enumerate_physical_devices()? }
            .iter()
            .find(|&&physical_device| {
                let props = unsafe { self.get_physical_device_properties(physical_device) };
                props.device_id == id
            })
            .copied();

        Ok(device)
    }

    fn find_physical_device(
        &self,
        required_extensions: &[&CStr],
    ) -> VkResult<Option<PhysicalDevice>> {
        let device = unsafe { self.enumerate_physical_devices() }?
            .into_iter()
            .find(|&physical_device| {
                unsafe { self.enumerate_device_extension_properties(physical_device) }
                    .map(|exts| {
                        let set: HashSet<&CStr> = exts
                            .iter()
                            .map(|ext| unsafe {
                                CStr::from_ptr(&ext.extension_name as *const c_char)
                            })
                            .collect();

                        required_extensions.iter().all(|ext| set.contains(ext))
                    })
                    .unwrap_or(false)
            });

        Ok(device)
    }
}

fn luxel_to_world<'a>(face: &Face, bsp: &'a Bsp<'a>, s: f32, t: f32) -> Vec3 {
    use lump_definitions::source::{Plane, TextureInfo};

    let plane = <Face as Associated<Plane>>::associated(face, bsp);
    let tex = <Face as Associated<TextureInfo>>::associated(face, bsp);

    let s_luxels = Vec3::from_array(tex.luxels[0].xyz);
    let t_luxels = Vec3::from_array(tex.luxels[1].xyz);

    let cross = t_luxels.cross(s_luxels);

    let normal = Vec3::from_array(plane.normal);
    let det = -normal.dot(cross);
    if det.abs() < 1.0e-20 {
        warn!("face vectors parallel to face normal");
    }

    let luxel_to_world = [t_luxels.cross(normal) / det, normal.cross(s_luxels) / det];

    let luxel_origin = -(plane.dist * cross) / det
        + luxel_to_world[0] * -tex.luxels[0].offset
        + luxel_to_world[1] * -tex.luxels[1].offset;

    let [s_min, t_min] = face.lightmap.mins;
    let s = s + s_min as f32;
    let t = t + t_min as f32;

    luxel_origin + luxel_to_world[0] * s + luxel_to_world[1] * t
}

fn run() -> Result<()> {
    let args = Args::parse();

    let contents = std::fs::read(args.bsp_path)?;
    let bsp = Bsp::parse(&contents).map_err(|_| anyhow!("failed to parse BSP file"))?;

    // NOTE: we can't call any vulkan functions after this is dropped.
    let entry = unsafe { Entry::load() }?;

    let mut instance_layers = vec![];
    if args.validation_layer {
        instance_layers.push(c"VK_LAYER_KHRONOS_validation")
    }

    let instance = create_vk_instance(&entry, instance_layers)?;
    let extensions = &[
        acceleration_structure::NAME,
        deferred_host_operations::NAME,
        ray_tracing_pipeline::NAME,
        spirv_1_4::NAME,
        scalar_block_layout::NAME,
        get_memory_requirements2::NAME,
    ];

    let physical_device = if let Some(id) = args.device_id {
        instance.physical_device_by_id(id)?
    } else {
        instance.find_physical_device(extensions)?
    }
    .context("Failed to find physical device")?;

    let physical_device_properties =
        unsafe { instance.get_physical_device_properties(physical_device) };
    info!(
        "Selected device '{}' ({})",
        physical_device_properties
            .device_name_as_c_str()
            .map(CStr::to_string_lossy)
            .unwrap_or(Cow::Borrowed("unknown")),
        physical_device_properties.device_id
    );

    let queue_family_idx = physical_device
        .find_queue_family_idx(&instance, |prop| {
            prop.queue_count > 0 && prop.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .context("Failed to find queue family index")?;

    let device: Device = {
        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_idx)
            .queue_priorities(&[1.0])];

        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut features2);
        };

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .vulkan_memory_model(true);

        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);

        let mut raytracing_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

        let enabled_extension_names = extensions.iter().map(|c| c.as_ptr()).collect::<Vec<_>>();

        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut features2)
            .push_next(&mut features12)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut raytracing_pipeline_features)
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(enabled_extension_names.as_slice());

        unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .context("Failed to create device")?
    };

    let device_queue = unsafe { device.get_device_queue(queue_family_idx, 0) };

    let command_pool = {
        let command_pool_create_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_idx);

        unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .context("Failed to create command pool")?
    };

    let device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let vk_ctx = VkContext::new(
        &device,
        &device_queue,
        &command_pool,
        &device_memory_properties,
    )?;

    let as_device = acceleration_structure::Device::new(&instance, &device);
    let rt_pipeline_device = ray_tracing_pipeline::Device::new(&instance, &device);
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
        .map_err(|_| anyhow!("Failed to get vertices lump"))?;

    let mut vertex_buffer = Buffer::new(
        &vk_ctx,
        (size_of::<Vertex>() * vertices.len()) as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    vertex_buffer.store(&vertices);

    let faces = bsp
        .lump_cast::<[Face], _>(LumpDefinition::Faces)
        .map_err(|_| anyhow!("Failed to get faces lump"))?;

    let mut texels: Vec<TexelData> = Vec::new();

    for face in faces.iter() {
        let plane = <Face as Associated<Plane>>::associated(face, &bsp);
        let normal = Vec3::from_array(plane.normal).normalize();

        let width = (face.lightmap.maxs[0] + 1) as u32;
        let height = (face.lightmap.maxs[1] + 1) as u32;

        for t in 0..height {
            for s in 0..width {
                let world_pos = luxel_to_world(face, &bsp, s as f32, t as f32);
                texels.push(TexelData::new(world_pos, normal));
            }
        }
    }

    let mut texel_buffer = Buffer::new(
        &vk_ctx,
        (size_of::<TexelData>() * texels.len()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    texel_buffer.store(&texels);

    let mut lighting_buffer = Buffer::new(
        &vk_ctx,
        (size_of::<AlignedVec3>() * texels.len()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    lighting_buffer.store(vec![AlignedVec3::default(); texels.len()].as_slice());

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

    let index_capacity = faces
        .iter()
        .map(|face| {
            let surface_edges = <Face as Associated<[SurfaceEdge]>>::associated(face, &bsp);
            // n edges produces (n - 2) triangles, each triangle has 3 indices
            let triangle_count = surface_edges.len().saturating_sub(2);
            triangle_count * 3
        })
        .sum();

    let mut indices = Vec::with_capacity(index_capacity);

    dbg!((indices.capacity(), indices.len()));
    for face in faces.iter() {
        let texture_data_idx =
            texture_info_lump[face.texture_info_index as usize].texture_data_index;
        let table_idx = texture_data_lump[texture_data_idx as usize].name_index;
        let data_idx = texture_data_string_table[table_idx as usize];
        let texture_name =
            CStr::from_bytes_until_nul(&texture_data_string_data[data_idx as usize..])
                .unwrap()
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

    dbg!((indices.capacity(), indices.len()));

    let mut index_buffer = Buffer::new(
        &vk_ctx,
        (size_of::<u16>() * indices.len()) as u64,
        vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    index_buffer.store(&indices);

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
                .index_type(vk::IndexType::UINT16),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let blas = AccelerationStructure::build(
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
                device_handle: blas.device_address(),
            },
        }];
        let instance_buffer_size =
            size_of::<vk::AccelerationStructureInstanceKHR>() * instances.len();

        let mut instance_buffer = Buffer::new(
            &vk_ctx,
            instance_buffer_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        instance_buffer.store(&instances);

        instance_buffer
    };

    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: instance_buffer.device_address(),
        });

    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances });

    let tlas = AccelerationStructure::build(
        &vk_ctx,
        &as_device,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        &[geometry],
        &[vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1)],
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
    )?;

    let (descriptor_set_layout, graphics_pipeline, pipeline_layout, shader_group_count) = {
        let binding_flags_inner = [
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
            vk::DescriptorBindingFlagsEXT::empty(),
        ];

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
            .binding_flags(&binding_flags_inner);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
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
            p_code: SHADER.as_ptr().cast::<u32>(),
            ..Default::default()
        };

        let shader_module =
            unsafe { device.create_shader_module(&shader_module_create_info, None) }?;

        let layouts = vec![descriptor_set_layout];
        let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_create_info, None) }?;

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

    let handle_size = rt_pipeline_properties.shader_group_handle_size as usize;
    let handle_alignment = rt_pipeline_properties.shader_group_base_alignment as usize;

    let handle_size_aligned = (handle_size + handle_alignment - 1) & !(handle_alignment - 1);

    let shader_binding_table_buffer = {
        let incoming_table_data = unsafe {
            rt_pipeline_device.get_ray_tracing_shader_group_handles(
                graphics_pipeline,
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
            &vk_ctx,
            table_size as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        shader_binding_table_buffer.store(&table_data);

        shader_binding_table_buffer
    };

    let descriptor_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        descriptor_count: 1,
    }];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&descriptor_sizes)
        .max_sets(1);

    let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }?;

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
        device.update_descriptor_sets(&[accel_write, texel_write, lighting_write], &[]);
    }

    {
        let sbt_address = shader_binding_table_buffer.device_address();

        let handle_size_aligned = handle_size_aligned as u64;
        let sbt_raygen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address)
            .size(handle_size_aligned)
            .stride(handle_size_aligned);

        let sbt_hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + handle_size_aligned)
            .size(handle_size_aligned)
            .stride(handle_size_aligned);

        let sbt_miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + 2 * handle_size_aligned)
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
                texels.len() as u32,
                1,
                1,
            );

            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(lighting_buffer.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );

            device.end_command_buffer(command_buffer)?;
        }
    }

    unsafe {
        device
            .queue_submit(
                device_queue,
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                vk::Fence::null(),
            )
            .context("Failed to execute queue submit.")?;

        device.queue_wait_idle(device_queue)?;
    }

    let lighting: Vec<AlignedVec3> = lighting_buffer.load(texels.len());
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

    bsp.write_to_io(&mut File::create("out_radiosity.bsp")?)
        .context("writing to io failed")?;

    unsafe {
        device.destroy_command_pool(command_pool, None);
    }

    unsafe {
        device.destroy_pipeline(graphics_pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
    }

    blas.destroy();
    tlas.destroy();

    shader_binding_table_buffer.destroy();
    texel_buffer.destroy();
    lighting_buffer.destroy();
    instance_buffer.destroy();
    vertex_buffer.destroy();
    index_buffer.destroy();

    unsafe {
        device.destroy_device(None);
    }

    unsafe {
        instance.destroy_instance(None);
    }

    Ok(())
}
fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(LevelFilter::Warn)
        .filter_module("radiosity", LevelFilter::Info)
        .parse_env("RUST_LOG")
        .format_timestamp(None)
        .init();

    if let Err(e) = run() {
        e.chain().rev().for_each(|e| error!("{e}"));
        std::process::exit(1);
    }

    Ok(())
}
