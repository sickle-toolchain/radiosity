#![feature(asm_experimental_arch)]
#![allow(unexpected_cfgs)]
#![cfg_attr(target_arch = "spirv", no_std)]

mod light;

use spirv_std::glam::{UVec3, Vec3};
use spirv_std::ray_tracing::AccelerationStructure;
use spirv_std::spirv;

use shared::{AlignedVec3, Light, Sky, TexelData};

const SAMPLES_PER_LUXEL: u32 = 16;

fn texel_index(gid: UVec3) -> usize {
    (gid.y * shared::COMPUTE_X_STRIDE + gid.x) as usize
}

#[spirv(compute(threads(64)))]
pub fn compute_sky(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] texels: &[TexelData],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] output: &mut [AlignedVec3],
    #[spirv(descriptor_set = 0, binding = 4, storage_buffer)] sky: &Sky,
) {
    let idx = texel_index(gid);
    if idx >= texels.len() {
        return;
    }

    let texel = texels[idx];
    let normal = texel.normal.0;

    let mut result = Vec3::ZERO;
    for s in 0..SAMPLES_PER_LUXEL {
        let sample_pos = light::jittered_position(&texel, s, SAMPLES_PER_LUXEL);
        result += light::contribute_sky(sky, sample_pos, normal, tlas, s, SAMPLES_PER_LUXEL);
    }

    output[idx] = AlignedVec3(result * (1.0 / SAMPLES_PER_LUXEL as f32));
}

#[spirv(compute(threads(64)))]
pub fn compute_world(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] texels: &[TexelData],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] output: &mut [AlignedVec3],
    #[spirv(descriptor_set = 0, binding = 3, storage_buffer)] lights: &[Light],
) {
    let idx = texel_index(gid);
    if idx >= texels.len() {
        return;
    }

    let texel = texels[idx];
    let normal = texel.normal.0;

    let mut result = Vec3::ZERO;
    for s in 0..SAMPLES_PER_LUXEL {
        let sample_pos = light::jittered_position(&texel, s, SAMPLES_PER_LUXEL);
        for i in 0..lights.len() {
            result += light::contribute_positional(&lights[i], sample_pos, normal, tlas);
        }
    }

    output[idx] = AlignedVec3(output[idx].0 + result * (1.0 / SAMPLES_PER_LUXEL as f32));
}
