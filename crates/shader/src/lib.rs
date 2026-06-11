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

use shared::{AlignedVec3, Light, Sky, TexelData};

const SAMPLES_PER_LUXEL: u32 = 16;

#[repr(u32)]
pub enum HitKind {
    Miss,
    Hit,
    Sky,
}

pub type RayPayload = HitKind;

#[spirv(miss)]
pub fn miss(#[spirv(incoming_ray_payload)] payload: &mut RayPayload) {
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
pub fn ray_generation_sky(
    #[spirv(launch_id)] launch_id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] texels: &[TexelData],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] output: &mut [AlignedVec3],
    #[spirv(descriptor_set = 0, binding = 4, storage_buffer)] sky: &Sky,
    #[spirv(ray_payload)] payload: &mut RayPayload,
) {
    let idx = launch_id.x as usize;

    if idx >= texels.len() {
        return;
    }

    let texel = texels[idx];
    let normal = texel.normal.0;

    let mut result = Vec3::ZERO;
    for s in 0..SAMPLES_PER_LUXEL {
        let sample_pos = light::jittered_position(&texel, s, SAMPLES_PER_LUXEL);
        result +=
            light::contribute_sky(sky, sample_pos, normal, tlas, payload, s, SAMPLES_PER_LUXEL);
    }

    output[idx] = AlignedVec3(result * (1.0 / SAMPLES_PER_LUXEL as f32));
}

#[spirv(ray_generation)]
pub fn ray_generation_world(
    #[spirv(launch_id)] launch_id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] texels: &[TexelData],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] output: &mut [AlignedVec3],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] lights: &[Light],
    #[spirv(ray_payload)] payload: &mut RayPayload,
) {
    let idx = launch_id.x as usize;

    if idx >= texels.len() {
        return;
    }

    let texel = texels[idx];
    let normal = texel.normal.0;

    let mut result = Vec3::ZERO;
    for s in 0..SAMPLES_PER_LUXEL {
        let sample_pos = light::jittered_position(&texel, s, SAMPLES_PER_LUXEL);
        for i in 0..lights.len() {
            result += light::contribute_positional(&lights[i], sample_pos, normal, tlas, payload);
        }
    }

    output[idx] = AlignedVec3(output[idx].0 + result * (1.0 / SAMPLES_PER_LUXEL as f32));
}

#[spirv(ray_generation)]
pub fn ray_generation_gi(
    #[spirv(launch_id)] _launch_id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] _tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] _texels: &[TexelData],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] _output: &mut [AlignedVec3],
    #[spirv(ray_payload)] _payload: &mut RayPayload,
) {
    black_box(())
}
