#![feature(asm_experimental_arch)]
#![allow(unexpected_cfgs)]
#![cfg_attr(target_arch = "spirv", no_std)]

// rust-analyzer keeps saying this is unused
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use core::f32::consts::PI;

use spirv_std::glam::{UVec3, Vec3, vec3};
use spirv_std::ray_tracing::{AccelerationStructure, RayFlags};
use spirv_std::spirv;

use shared::{AlignedVec3, TexelData};

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    x
}

fn rand(seed: &mut u32) -> f32 {
    *seed = hash(*seed);
    (*seed as f32) / (u32::MAX as f32)
}

fn cosine_hemisphere(seed: &mut u32) -> Vec3 {
    let r1 = rand(seed);
    let r2 = rand(seed);

    let phi = 2.0 * PI * r1;
    let r = r2.sqrt();

    vec3(phi.cos() * r, phi.sin() * r, (1.0 - r2).sqrt())
}

fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.z.abs() < 0.999 {
        vec3(0.0, 0.0, 1.0)
    } else {
        vec3(1.0, 0.0, 0.0)
    };

    let t = n.cross(up).normalize();
    let b = n.cross(t);
    (t, b)
}

#[spirv(miss)]
pub fn miss(#[spirv(incoming_ray_payload)] radiance: &mut Vec3) {
    *radiance = vec3(1.0, 1.0, 1.0);
}

#[spirv(closest_hit)]
pub fn closest_hit(#[spirv(incoming_ray_payload)] radiance: &mut Vec3) {
    *radiance = vec3(0.05, 0.05, 0.05);
}

#[spirv(ray_generation)]
pub fn ray_generation(
    #[spirv(launch_id)] launch_id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] texels: &[TexelData],
    #[spirv(descriptor_set = 0, binding = 2, storage_buffer)] lighting: &mut [AlignedVec3],
    #[spirv(ray_payload)] incoming: &mut Vec3,
) {
    let texel_index = launch_id.x as usize;
    let texel = &texels[texel_index];

    let normal: Vec3 = texel.normal.into();
    let position: Vec3 = texel.position.into();

    let mut seed = texel_index as u32 * 9781 + 1;

    let (tangent, bitangent) = tangent_basis(normal);

    let mut accumulated = Vec3::ZERO;
    let samples = 64;

    for _ in 0..samples {
        let local_dir = cosine_hemisphere(&mut seed);

        let world_dir =
            (tangent * local_dir.x + bitangent * local_dir.y + normal * local_dir.z).normalize();

        *incoming = Vec3::ZERO;

        unsafe {
            tlas.trace_ray(
                RayFlags::CULL_BACK_FACING_TRIANGLES,
                0xff,
                0,
                0,
                0,
                position + normal * 0.01,
                0.001,
                world_dir,
                1.0e30,
                incoming,
            );
        }

        accumulated += *incoming;
    }

    lighting[texel_index] = (accumulated / samples as f32).into();
}
