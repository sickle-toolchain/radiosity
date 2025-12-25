use std::borrow::Cow;
use std::ffi::CStr;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::ptr;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use log::{LevelFilter, error, info, warn};
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
    pub timestamp_query_pool: QueryPool,
}

fn format_duration_ms(ms: f64) -> String {
    let mut remaining_ms = ms.max(0.0);
    let hours = (remaining_ms / 3_600_000.0).floor() as u64;
    remaining_ms -= hours as f64 * 3_600_000.0;

    let minutes = (remaining_ms / 60_000.0).floor() as u64;
    remaining_ms -= minutes as f64 * 60_000.0;

    let seconds = (remaining_ms / 1_000.0).floor() as u64;
    remaining_ms -= seconds as f64 * 1_000.0;

    let mut parts = Vec::new();
    if hours > 0 {
        parts.push(format!("{hours}h"));
    }
    if minutes > 0 {
        parts.push(format!("{minutes}m"));
    }

    if seconds > 0 {
        let total_seconds = seconds as f64 + remaining_ms / 1000.0;
        parts.push(format!("{total_seconds:.3}s"));
    } else {
        parts.push(format!("{remaining_ms:.3}ms"));
    }

    parts.join(" ")
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

        Ok(Self {
            ctx,
            acceleration_structure_device,
            ray_tracing_pipeline_device,
            ray_tracing_pipeline_properties,
            timestamp_query_pool,
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

    fn read_timestamp_ms(&self) -> Result<f64> {
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
        Ok(time_ns / 1_000_000.0)
    }

    // Create triangular acceleration structure geometry
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
        const MEMORY_PROPERTYIES: MemoryPropertyFlags = MemoryPropertyFlags::from_raw(
            MemoryPropertyFlags::HOST_VISIBLE.as_raw()
                | MemoryPropertyFlags::HOST_COHERENT.as_raw(),
        );

        let mut vertex_buffer = Buffer::new(
            self.ctx.clone(),
            size_of_val(vertices) as u64,
            INPUT_BUFFER_FLAGS | BufferUsageFlags::VERTEX_BUFFER,
            MEMORY_PROPERTYIES,
        )?;
        vertex_buffer.store(vertices);

        let mut index_buffer = Buffer::new(
            self.ctx.clone(),
            size_of_val(indices) as u64,
            INPUT_BUFFER_FLAGS | BufferUsageFlags::INDEX_BUFFER,
            MEMORY_PROPERTYIES,
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
                    .index_type(vk::IndexType::UINT16),
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        Ok((geometry, vertex_buffer, index_buffer))
    }

    // Create instance acceleration structure geometry
    fn create_instance_geometry(
        &'_ self,
        instance: AccelerationStructureKHR,
        matrix: [f32; 12],
    ) -> Result<(AccelerationStructureGeometryKHR<'_>, Buffer)> {
        const INPUT_BUFFER_FLAGS: BufferUsageFlags = BufferUsageFlags::from_raw(
            BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw()
                | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR.as_raw(),
        );
        const MEMORY_PROPERTYIES: MemoryPropertyFlags = MemoryPropertyFlags::from_raw(
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
            (size_of::<vk::AccelerationStructureInstanceKHR>() * instances.len()) as vk::DeviceSize,
            INPUT_BUFFER_FLAGS,
            MEMORY_PROPERTYIES,
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
                .command_pool(self.ctx.pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers =
                unsafe { self.ctx.device.allocate_command_buffers(&allocate_info) }.unwrap();
            command_buffers[0]
        };

        unsafe {
            self.ctx
                .device
                .begin_command_buffer(
                    build_command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

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
            self.ctx
                .device
                .end_command_buffer(build_command_buffer)
                .unwrap();
            self.ctx.device.queue_submit(
                self.ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&[build_command_buffer])],
                vk::Fence::null(),
            )?;

            self.ctx.device.queue_wait_idle(self.ctx.queue).unwrap();
            self.ctx
                .device
                .free_command_buffers(self.ctx.pool, &[build_command_buffer]);
        }

        let build_time_ms = self.read_timestamp_ms()?;
        info!(
            "Built {structure_type:#?} acceleration structure in {}",
            format_duration_ms(build_time_ms)
        );

        Ok((acceleration_structure, buffer))
    }
}

impl Drop for Application<'_> {
    fn drop(&mut self) {
        unsafe {
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
    let app = Application::new(ctx.clone())?;

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

    let mut texel_buffer = Buffer::new(
        ctx.clone(),
        (size_of::<TexelData>() * texels.len()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    texel_buffer.store(&texels);

    let mut lighting_buffer = Buffer::new(
        ctx.clone(),
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

    if let Some(obj_path) = &args.dump_blas_geometry {
        let mut obj_file = File::create(obj_path).context("Failed to create OBJ file")?;

        // Write vertices
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

    let (blas_geometry, _vertex_buffer, _index_buffer) =
        app.create_triangle_geometry(&vertices, &indices)?;
    let (blas, _blas_buffer) = app.create_as(
        AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        &[blas_geometry],
        &[vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(indices.len() as u32 / 3)],
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
    )?;

    let identity_matrix: [f32; 12] = std::array::from_fn(|i| Mat4::IDENTITY.to_cols_array()[i]);
    let (tlas_geometry, _tlas_geometry_buffer) =
        app.create_instance_geometry(blas, identity_matrix)?;

    let (tlas, _tlas_buffer) = app.create_as(
        AccelerationStructureTypeKHR::TOP_LEVEL,
        &[tlas_geometry],
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
            ctx.device.create_descriptor_set_layout(
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

        let shader_module = unsafe {
            ctx.device
                .create_shader_module(&shader_module_create_info, None)
        }?;

        let layouts = vec![descriptor_set_layout];
        let layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

        let pipeline_layout =
            unsafe { ctx.device.create_pipeline_layout(&layout_create_info, None) }?;

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
            app.ray_tracing_pipeline_device
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
            ctx.device.destroy_shader_module(shader_module, None);
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
            .command_pool(ctx.pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            ctx.device
                .allocate_command_buffers(&command_buffer_allocate_info)
        }?[0]
    };

    {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

        unsafe {
            ctx.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
        }?;
    }

    let handle_size = app.ray_tracing_pipeline_properties.shader_group_handle_size as usize;
    let handle_alignment = app
        .ray_tracing_pipeline_properties
        .shader_group_base_alignment as usize;

    let handle_size_aligned = (handle_size + handle_alignment - 1) & !(handle_alignment - 1);

    let shader_binding_table_buffer = {
        let incoming_table_data = unsafe {
            app.ray_tracing_pipeline_device
                .get_ray_tracing_shader_group_handles(
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
            ctx.clone(),
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

    let descriptor_pool = unsafe {
        ctx.device
            .create_descriptor_pool(&descriptor_pool_info, None)
    }?;

    let mut count_allocate_info =
        vk::DescriptorSetVariableDescriptorCountAllocateInfo::default().descriptor_counts(&[1]);

    let descriptor_set = unsafe {
        ctx.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_set_layout])
                .push_next(&mut count_allocate_info),
        )
    }?[0];

    let accel_structs = [tlas];
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
        ctx.device
            .update_descriptor_sets(&[accel_write, texel_write, lighting_write], &[]);
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
            ctx.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                graphics_pipeline,
            );
            ctx.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            app.begin_timestamp_range(
                command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );

            app.ray_tracing_pipeline_device.cmd_trace_rays(
                command_buffer,
                &sbt_raygen_region,
                &sbt_miss_region,
                &sbt_hit_region,
                &sbt_call_region,
                texels.len() as u32,
                1,
                1,
            );

            app.end_timestamp_range(
                command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            );

            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(lighting_buffer.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE);

            ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );

            ctx.device.end_command_buffer(command_buffer)?;
        }
    }

    unsafe {
        ctx.device
            .queue_submit(
                ctx.queue,
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                vk::Fence::null(),
            )
            .context("Failed to execute queue submit.")?;

        ctx.device.queue_wait_idle(ctx.queue)?;
    }

    info!(
        "Radiosity shader executed in {}",
        format_duration_ms(app.read_timestamp_ms()?)
    );

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
        ctx.device.destroy_pipeline(graphics_pipeline, None);
        ctx.device.destroy_pipeline_layout(pipeline_layout, None);
        ctx.device
            .destroy_descriptor_set_layout(descriptor_set_layout, None);
        ctx.device.destroy_descriptor_pool(descriptor_pool, None);
    }

    unsafe {
        app.acceleration_structure_device
            .destroy_acceleration_structure(blas, None);
    }

    unsafe {
        app.acceleration_structure_device
            .destroy_acceleration_structure(tlas, None);
    }

    Ok(())
}
fn main() {
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
}
