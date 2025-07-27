#![feature(asm_experimental_arch)]
#![allow(unexpected_cfgs)]
#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::{
    glam::{UVec3, Vec2, Vec3, vec2, vec3},
    ray_tracing::{AccelerationStructure, RayFlags},
    spirv,
};

#[spirv(miss)]
pub fn miss(#[spirv(incoming_ray_payload)] out: &mut Vec3) {
    *out = vec3(0.5, 0.5, 0.5);
}

#[spirv(closest_hit)]
pub fn closest_hit(#[spirv(incoming_ray_payload)] out: &mut Vec3, #[spirv(instance_id)] id: u32) {
    *out = vec3((id & 1) as f32, (id & 2) as f32, (id & 4) as f32)
}

#[spirv(ray_generation)]
pub fn ray_generation(
    #[spirv(launch_id)] launch_id: UVec3,
    #[spirv(launch_size)] launch_size: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] top_level_as: &AccelerationStructure,
    // #[spirv(descriptor_set = 0, binding = 1)] image: &Image!(2D, format = rgba8, sampled = false),
    #[spirv(ray_payload)] payload: &mut Vec3,
) {
    let pixel_center = vec2(launch_id.x as f32, launch_id.y as f32) + vec2(0.5, 0.5);
    let in_uv = pixel_center / vec2(launch_size.x as f32, launch_size.y as f32);

    let d = in_uv * 2.0 - Vec2::ONE;
    let aspect_ratio = launch_size.x as f32 / launch_size.y as f32;

    let origin = vec3(0.0, 0.0, -50000.0);
    let direction = vec3(d.x * aspect_ratio, -d.y, 1.0).normalize();
    let cull_mask = 0xff;
    let tmin = 0.001;
    let tmax = 100000.0;

    *payload = Vec3::ZERO;

    unsafe {
        top_level_as.trace_ray(
            RayFlags::OPAQUE,
            cull_mask,
            0,
            0,
            0,
            origin,
            tmin,
            direction,
            tmax,
            payload,
        );
    }

    // unsafe {
    //     image.write(uvec2(launch_id.x, launch_id.y), payload.extend(1.0));
    // }
}
