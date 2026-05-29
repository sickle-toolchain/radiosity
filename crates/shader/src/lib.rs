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
    let result = light::contribute_sky(sky, texel.position.0, texel.normal.0, tlas, payload);

    output[idx] = AlignedVec3(result);
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
    let sample_pos: Vec3 = texel.position.0;
    let sample_normal: Vec3 = texel.normal.0;

    let mut result = output[idx].0;

    for i in 0..lights.len() {
        result +=
            light::contribute_positional(&lights[i], sample_pos, sample_normal, tlas, payload);
    }

    output[idx] = AlignedVec3(result);
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
