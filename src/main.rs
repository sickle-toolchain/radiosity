use std::borrow::Cow;
use std::cell::{Ref, RefMut};
use std::fs::File;
use std::path::PathBuf;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use tracing::{error, info, instrument, warn};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_timing::TreeTimingLayer;
use zerocopy::IntoBytes;

use ash::khr::{
    acceleration_structure, deferred_host_operations, get_memory_requirements2,
    ray_tracing_pipeline, spirv_1_4,
};
use ash::vk;
use spirv_std::glam::{Mat3, Vec3};

use bsp::Bsp;
use lump_definitions::source::{
    Brush, BrushSide, ColorRGBExp32, EmitType, Face, Leaf, LumpDefinition, MASK_OPAQUE, Model,
    Node, Plane, SurfaceEdge, SurfaceFlags, TextureInfo, Vertex, WorldLight,
};

use radiosity::vulkan::{Buffer, VulkanContext};
use radiosity::{Application, Associated};
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

const NON_LIT_SURFACE_FLAGS: u16 = SurfaceFlags::SKY
    | SurfaceFlags::SKY2D
    | SurfaceFlags::NOLIGHT
    | SurfaceFlags::NODRAW
    | SurfaceFlags::HINT
    | SurfaceFlags::SKIP;

fn is_lightmapped<'a>(face: &Face, bsp: &'a Bsp<'a>) -> bool {
    if face.texture_info_index < 0 {
        return false;
    }
    let tex: Ref<'_, TextureInfo> = face.associated(bsp);
    (tex.flags as u16) & NON_LIT_SURFACE_FLAGS == 0
}

#[instrument(skip_all)]
fn generate_texels<'a>(bsp: &'a Bsp<'a>, faces: &[Face], lightmap_scale: u32) -> Vec<TexelData> {
    let mut texels = Vec::new();

    for face in faces {
        if !is_lightmapped(face, bsp) {
            continue;
        }

        let plane: Ref<'_, Plane> = face.associated(bsp);
        let normal = Vec3::from_array(plane.normal).normalize();

        let width = (face.lightmap.maxs[0] + 1) as u32 * lightmap_scale;
        let height = (face.lightmap.maxs[1] + 1) as u32 * lightmap_scale;

        let base_matrix = luxel_to_world_matrix(face, bsp);
        let inv_scale = 1.0 / lightmap_scale as f32;
        let matrix = Mat3::from_cols(
            base_matrix.col(0) * inv_scale,
            base_matrix.col(1) * inv_scale,
            base_matrix.col(2),
        );
        let tangent = matrix.col(0);
        let bitangent = matrix.col(1);

        for t in 0..height {
            for s in 0..width {
                let world_pos = matrix * Vec3::new(s as f32, t as f32, 1f32);
                texels.push(TexelData::new(world_pos, normal, tangent, bitangent));
            }
        }
    }

    texels
}

fn base_winding_into(normal: Vec3, dist: f32, out: &mut Vec<Vec3>) {
    let abs_n = normal.abs();
    let axis = if abs_n.x > abs_n.y && abs_n.x > abs_n.z {
        Vec3::Y
    } else if abs_n.y > abs_n.z {
        Vec3::Z
    } else {
        Vec3::X
    };

    let tangent = (axis - normal * normal.dot(axis)).normalize();
    let bitangent = normal.cross(tangent);
    let center = normal * dist;

    const HALF_SIZE: f32 = 65536.0;
    out.clear();
    out.extend([
        center + tangent * HALF_SIZE + bitangent * HALF_SIZE,
        center - tangent * HALF_SIZE + bitangent * HALF_SIZE,
        center - tangent * HALF_SIZE - bitangent * HALF_SIZE,
        center + tangent * HALF_SIZE - bitangent * HALF_SIZE,
    ]);
}

fn chop_winding_into(winding: &[Vec3], normal: Vec3, dist: f32, epsilon: f32, out: &mut Vec<Vec3>) {
    out.clear();
    if winding.is_empty() {
        return;
    }
    let n = winding.len();
    let mut prev = winding[n - 1];
    let mut prev_signed = prev.dot(normal) - dist;
    let mut prev_in = prev_signed >= -epsilon;
    for &curr in winding {
        let curr_signed = curr.dot(normal) - dist;
        let curr_in = curr_signed >= -epsilon;
        if prev_in != curr_in {
            let t = prev_signed / (prev_signed - curr_signed);
            out.push(prev + (curr - prev) * t);
        }
        if curr_in {
            out.push(curr);
        }
        prev = curr;
        prev_signed = curr_signed;
        prev_in = curr_in;
    }
}

fn emit_polygon(
    vertices: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    polygon: impl IntoIterator<Item = [f32; 3]>,
) {
    let base = vertices.len() as u32;
    vertices.extend(polygon);
    let count = vertices.len() as u32 - base;
    indices.extend((2..count).flat_map(|k| [base, base + k - 1, base + k]));
}

fn collect_worldspawn_brush_indices<'a>(
    bsp: &'a Bsp<'a>,
) -> Result<std::collections::HashSet<u32>> {
    let models: Ref<'_, [Model]> = bsp.get_lump(LumpDefinition::Models)?;
    let nodes: Ref<'_, [Node]> = bsp.get_lump(LumpDefinition::Nodes)?;
    let leaves: Ref<'_, [Leaf]> = bsp.get_lump(LumpDefinition::Leaves)?;
    let leaf_brushes: Ref<'_, [u16]> = bsp.get_lump(LumpDefinition::LeafBrushes)?;

    let worldspawn_root = models
        .first()
        .context("BSP has no models lump entries")?
        .head_node;

    let mut result = std::collections::HashSet::new();
    let mut stack = vec![worldspawn_root];
    while let Some(child) = stack.pop() {
        if child >= 0 {
            let node = &nodes[child as usize];
            stack.push(node.children[0]);
            stack.push(node.children[1]);
        } else {
            let leaf_idx = !child as usize;
            let leaf = &leaves[leaf_idx];
            let first = leaf.first_leaf_brush as usize;
            let count = leaf.num_leaf_brushes as usize;

            result.extend(
                leaf_brushes[first..first + count]
                    .iter()
                    .map(|&b| u32::from(b)),
            );
        }
    }
    Ok(result)
}

#[instrument(skip_all)]
fn collect_brush_geometry<'a>(bsp: &'a Bsp<'a>) -> Result<(Vec<[f32; 3]>, Vec<u32>)> {
    let brushes: Ref<'_, [Brush]> = bsp.get_lump(LumpDefinition::Brushes)?;
    let sides: Ref<'_, [BrushSide]> = bsp.get_lump(LumpDefinition::BrushSides)?;
    let planes: Ref<'_, [Plane]> = bsp.get_lump(LumpDefinition::Planes)?;
    let texinfos: Ref<'_, [TextureInfo]> = bsp.get_lump(LumpDefinition::TextureInfo)?;
    let worldspawn_brushes = collect_worldspawn_brush_indices(bsp)?;

    const CHOP_EPSILON: f32 = 0.1;
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let mut winding: Vec<Vec3> = Vec::new();
    let mut scratch: Vec<Vec3> = Vec::new();

    for (brush_idx, brush) in brushes.iter().enumerate() {
        if !worldspawn_brushes.contains(&(brush_idx as u32)) {
            continue;
        }
        if brush.contents & MASK_OPAQUE == 0 {
            continue;
        }
        let first = brush.first_side as usize;
        let count = brush.num_sides as usize;

        for i in 0..count {
            let side = &sides[first + i];
            if side.disp_info != 0 {
                continue;
            }
            if side.tex_info < 0 {
                continue;
            }
            let tex = &texinfos[side.tex_info as usize];
            if (tex.flags as u16) & SurfaceFlags::SKY != 0 {
                continue;
            }

            let plane = &planes[side.plane_num as usize];
            let normal = Vec3::from_array(plane.normal);
            base_winding_into(normal, plane.dist, &mut winding);

            for j in 0..count {
                if i == j {
                    continue;
                }
                let other = &sides[first + j];
                if other.bevel != 0 {
                    continue;
                }
                let other_plane = &planes[other.plane_num as usize];
                let inward_normal = -Vec3::from_array(other_plane.normal);
                let inward_dist = -other_plane.dist;
                chop_winding_into(
                    &winding,
                    inward_normal,
                    inward_dist,
                    CHOP_EPSILON,
                    &mut scratch,
                );
                std::mem::swap(&mut winding, &mut scratch);
                if winding.len() < 3 {
                    break;
                }
            }

            if winding.len() < 3 {
                continue;
            }

            emit_polygon(
                &mut vertices,
                &mut indices,
                winding.iter().map(|v| v.to_array()),
            );
        }
    }

    Ok((vertices, indices))
}

#[instrument(skip_all)]
fn collect_sky_geometry<'a>(
    bsp: &'a Bsp<'a>,
    faces: &[Face],
    vertices: &mut Vec<[f32; 3]>,
) -> Result<Vec<u32>> {
    const SKY_FLAGS: u16 = SurfaceFlags::SKY | SurfaceFlags::SKY2D;

    let positions: Ref<'_, [Vertex]> = bsp.get_lump(LumpDefinition::Vertices)?;
    let positions: &[[f32; 3]] = zerocopy::transmute_ref!(&*positions);

    let mut indices = Vec::new();
    for face in faces {
        let tex: Ref<'_, TextureInfo> = face.associated(bsp);
        if (tex.flags as u16) & SKY_FLAGS == 0 {
            continue;
        }

        let surface_edges: Ref<'_, [SurfaceEdge]> = face.associated(bsp);
        if surface_edges.len() < 3 {
            continue;
        }

        emit_polygon(
            vertices,
            &mut indices,
            surface_edges.iter().map(|surface_edge| {
                let edge = surface_edge.associated(bsp).edge;
                positions[edge[usize::from(surface_edge.edge_index < 0)] as usize]
            }),
        );
    }
    Ok(indices)
}

#[instrument(skip_all)]
fn collect_lights<'a>(bsp: &'a Bsp<'a>) -> Result<(Vec<shared::Light>, shared::Sky)> {
    const LUMP_INTENSITY_SCALE: f32 = 255.0;

    let entity_string = {
        let entity_lump = bsp.lump(LumpDefinition::Entities);
        let raw = String::from_utf8_lossy(&entity_lump.data);
        let end = raw.find('\0').unwrap_or(raw.len());
        raw[..end].to_owned()
    };
    let entities = match valve_kv::parse(&entity_string) {
        Ok(entities) => entities,
        Err(e) => {
            warn!("failed to parse entity lump: {e}");
            Vec::new()
        }
    };

    let mut ambient_override: Option<Vec3> = None;
    let mut sun_spread = 0.0_f32;
    if let Some(env) = entities
        .iter()
        .find(|ent| ent.properties.get("classname") == Some(&"light_environment"))
    {
        if let Some(ambient_str) = env.properties.get("_ambient") {
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
        if let Some(spread) = env.properties.get("SunSpreadAngle") {
            if let Ok(angle) = spread.parse::<f32>() {
                sun_spread = angle.to_radians().sin();
                info!("Sun angular radius (sin): {sun_spread}");
            }
        }
    }

    let mut world_lights: Vec<shared::Light> = Vec::new();
    let mut sky = shared::Sky::default();

    let worldlights = bsp
        .get_lump::<[WorldLight]>(LumpDefinition::WorldLights)
        .context("Failed to parse worldlights lump")?;

    for (i, wl) in worldlights.iter().enumerate() {
        let Ok(ty) = EmitType::try_from(wl.ty) else {
            warn!(
                "Light {} is an unsupported type ({}) and was skipped.",
                i, wl.ty
            );
            continue;
        };

        let color =
            Vec3::new(wl.intensity[0], wl.intensity[1], wl.intensity[2]) * LUMP_INTENSITY_SCALE;

        match ty {
            EmitType::SkyLight => {
                sky.sun_direction = Vec3::new(wl.normal[0], wl.normal[1], wl.normal[2]).into();
                sky.sun_color = color.into();
            }
            EmitType::SkyAmbient => {
                sky.ambient_color = (sky.ambient_color.0 + color).into();
            }
            EmitType::Point | EmitType::Spotlight | EmitType::Surface | EmitType::QuakeLight => {
                let mut c = wl.constant_attn;
                let mut l = wl.linear_attn;
                let mut q = wl.quadratic_attn;

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
                    EmitType::QuakeLight => {
                        c = 0.0;
                        l = 1.0;
                        q = 0.0;
                    }
                    _ => {}
                }

                let light = shared::Light {
                    position: Vec3::new(wl.origin[0], wl.origin[1], wl.origin[2]).into(),
                    color: color.into(),
                    direction: Vec3::new(wl.normal[0], wl.normal[1], wl.normal[2]).into(),
                    ty,
                    radius: wl.radius,
                    constant_attn: c,
                    linear_attn: l,
                    quadratic_attn: q,
                    penumbra_start: wl.penumbra_start,
                    penumbra_end: wl.penumbra_end,
                    exponent: wl.exponent,
                };

                info!("World light {i}: {light:?}");
                world_lights.push(light);
            }
        }
    }

    if let Some(ambient_color) = ambient_override {
        sky.ambient_color = ambient_color.into();
    }
    sky.sun_spread = sun_spread;

    Ok((world_lights, sky))
}

fn encode_rgbexp32(color: Vec3) -> ColorRGBExp32 {
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
}

#[instrument(skip_all)]
fn write_lightmap<'a>(
    bsp: &'a Bsp<'a>,
    lighting: &[AlignedVec3],
    lightmap_scale: u32,
) -> Result<()> {
    let mut faces = bsp.get_lump_mut::<[Face]>(LumpDefinition::Faces)?;
    let mut faces_hdr = bsp.get_lump_mut::<[Face]>(LumpDefinition::FacesHdr).ok();

    if lightmap_scale > 1 {
        let scale = lightmap_scale as f32;
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
            face.lightmap.mins[0] *= lightmap_scale as i32;
            face.lightmap.mins[1] *= lightmap_scale as i32;
        }
        if let Some(hdr) = &mut faces_hdr {
            for face in hdr.iter_mut() {
                face.lightmap.mins[0] *= lightmap_scale as i32;
                face.lightmap.mins[1] *= lightmap_scale as i32;
            }
        }
    }

    let mut lightmap_samples: Vec<ColorRGBExp32> = Vec::with_capacity(lighting.len() + faces.len());
    let mut byte_offset: i32 = 0;
    let mut texel_offset = 0usize;

    for (i, face) in faces.iter_mut().enumerate() {
        if !is_lightmapped(face, bsp) {
            face.light_offset = -1;
            if let Some(hdr_faces) = &mut faces_hdr
                && let Some(hdr_face) = hdr_faces.get_mut(i)
            {
                hdr_face.light_offset = -1;
            }
            continue;
        }

        let width = ((face.lightmap.maxs[0] + 1) as u32 * lightmap_scale) as usize;
        let height = ((face.lightmap.maxs[1] + 1) as u32 * lightmap_scale) as usize;
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
    let mut lighting_lump = bsp.lump_mut(LumpDefinition::Lighting);
    let mut lighting_hdr_lump = bsp.lump_mut(LumpDefinition::LightingHdr);
    lighting_lump.data = Cow::Owned(final_bytes.clone());
    lighting_hdr_lump.data = Cow::Owned(final_bytes);

    info!("Wrote {} bytes to lighting lump", lighting_lump.data.len());

    Ok(())
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

    let faces = bsp.get_lump::<[Face]>(LumpDefinition::Faces)?;

    let texels = generate_texels(&bsp, &faces, args.lightmap_scale);
    if texels.is_empty() {
        bail!("no texels");
    }

    let mut texel_staging = Buffer::new(
        ctx.clone(),
        (size_of::<TexelData>() * texels.len()) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    texel_staging.store(&texels);

    let command_buffer = app.command_buffer;

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        ctx.device
            .begin_command_buffer(command_buffer, &begin_info)?;
    }

    let texel_buffer = Buffer::new(
        ctx.clone(),
        texel_staging.size(),
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    texel_buffer.cmd_copy_from(command_buffer, &texel_staging, texel_staging.size());

    let output_buffer = Buffer::new(
        ctx.clone(),
        (size_of::<AlignedVec3>() * texels.len()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    let output_readback = Buffer::new(
        ctx.clone(),
        output_buffer.size(),
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let (mut geometry_vertices, solid_indices) = collect_brush_geometry(&bsp)?;
    let sky_indices = collect_sky_geometry(&bsp, &faces, &mut geometry_vertices)?;
    info!(
        "Geometry: {} solid triangles, {} sky triangles",
        solid_indices.len() / 3,
        sky_indices.len() / 3
    );

    let setup_buffers = app.create_acceleration_structures(
        command_buffer,
        &geometry_vertices,
        &solid_indices,
        &sky_indices,
    )?;

    let (world_lights, sky) = collect_lights(&bsp)?;

    let mut sky_staging = Buffer::new(
        ctx.clone(),
        size_of::<shared::Sky>() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    sky_staging.store(std::slice::from_ref(&sky));

    let sky_buffer = Buffer::new(
        ctx.clone(),
        sky_staging.size(),
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    sky_buffer.cmd_copy_from(command_buffer, &sky_staging, sky_staging.size());

    let world_bytes = (size_of::<shared::Light>() * world_lights.len()).max(1) as vk::DeviceSize;
    let mut world_staging = Buffer::new(
        ctx.clone(),
        world_bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let world_buffer = Buffer::new(
        ctx.clone(),
        world_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    if !world_lights.is_empty() {
        world_staging.store(&world_lights);
        world_buffer.cmd_copy_from(command_buffer, &world_staging, world_bytes);
    }

    let init_to_rt_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags2::ALL_TRANSFER,
        )
        .src_access_mask(
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::TRANSFER_WRITE,
        )
        .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
        .dst_access_mask(
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
        );

    unsafe {
        ctx.device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().memory_barriers(&[init_to_rt_barrier]),
        );
    }

    let shader_group_count = app.create_pipeline()?;
    let sbt_handle_size = app.create_shader_binding_table(shader_group_count)?;

    app.create_descriptor_set(&texel_buffer, &output_buffer, &sky_buffer, &world_buffer)?;
    app.record_ray_tracing(
        sbt_handle_size,
        texels.len(),
        &output_buffer,
        Some(&output_readback),
    )?;

    drop(setup_buffers);

    let lighting: Vec<AlignedVec3> = output_readback.load(texels.len());

    // Drop immutable ref so we can take mutable ref
    drop(faces);
    write_lightmap(&bsp, &lighting, args.lightmap_scale)?;

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
