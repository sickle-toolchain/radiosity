#![feature(asm_experimental_arch)]
#![allow(unexpected_cfgs)]
#![cfg_attr(target_arch = "spirv", no_std)]

mod light;

use core::hint::black_box;

use spirv_std::glam::{UVec3, Vec3};
use spirv_std::ray_tracing::AccelerationStructure;
use spirv_std::spirv;

#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use shared::{AlignedVec3, EmitType, Light, TexelData};

#[repr(u32)]
pub enum HitKind {
    Miss,
    Hit,
    Sky,
}

pub type RayPayload = HitKind;

#[spirv(miss)]
pub fn miss(#[spirv(incoming_ray_payload)] payload: &mut RayPayload) {
    // TODO: Can we just remove the miss function?
    *payload = HitKind::Miss;
    black_box(())
}

#[spirv(closest_hit)]
pub fn closest_hit(#[spirv(incoming_ray_payload)] payload: &mut RayPayload) {
    *payload = HitKind::Hit;
    black_box(());
}

#[spirv(closest_hit)]
pub fn sky_hit(#[spirv(incoming_ray_payload)] payload: &mut RayPayload) {
    *payload = HitKind::Sky;
    black_box(());
}

#[spirv(ray_generation)]
pub fn ray_generation_direct(
    #[spirv(launch_id)] launch_id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] texels: &[TexelData],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] lighting: &mut [AlignedVec3],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] lights: &[Light],
    #[spirv(ray_payload)] payload: &mut RayPayload,
) {
    let idx = launch_id.x as usize;

    if idx >= texels.len() {
        return;
    }

    let texel = texels[idx];
    let sample_pos: Vec3 = texel.position.0;
    let sample_normal: Vec3 = texel.normal.0;

    let mut result = Vec3::ZERO;

    for i in 0..lights.len() {
        let light = &lights[i];

        result += match light.ty {
            EmitType::SkyAmbient => light::contribute_sky_ambient(light),
            EmitType::SkyLight => {
                light::contribute_sky_light(light, sample_pos, sample_normal, tlas, payload)
            }
            EmitType::Point | EmitType::Spotlight | EmitType::Surface | EmitType::QuakeLight => {
                light::contribute_positional(light, sample_pos, sample_normal, tlas, payload)
            }
        };
    }

    lighting[idx] = AlignedVec3(result);
}

#[spirv(ray_generation)]
pub fn ray_generation_gi(
    #[spirv(launch_id)] _launch_id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] _tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] _texels: &[TexelData],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] _lighting: &mut [AlignedVec3],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] _lights: &[Light],
    #[spirv(ray_payload)] _payload: &mut RayPayload,
) {
    black_box(())
}
