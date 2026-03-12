use spirv_std::glam::Vec3;
use spirv_std::ray_tracing::{AccelerationStructure, RayFlags};

#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use shared::{EmitType, Light};

use crate::{HitKind, RayPayload};

pub const SHADOW_EPSILON: f32 = 0.25;
pub const INTENSITY_SCALE: f32 = 255.0;

#[inline(always)]
pub fn contribute_sky_ambient(light: &Light) -> Vec3 {
    light.color
}

#[inline(always)]
pub fn contribute_sky_light(
    light: &Light,
    sample_pos: Vec3,
    sample_normal: Vec3,
    tlas: &AccelerationStructure,
    payload: &mut RayPayload,
) -> Vec3 {
    let sun_dir = -light.direction;
    let n_dot_l = sample_normal.dot(sun_dir);
    if n_dot_l <= 0.0 {
        return Vec3::ZERO;
    }

    let ray_origin = sample_pos + sample_normal * SHADOW_EPSILON;
    *payload = HitKind::Miss;

    unsafe {
        tlas.trace_ray(
            RayFlags::OPAQUE,
            0xFF,
            0,
            0,
            0,
            ray_origin,
            0.001,
            sun_dir,
            100000.0,
            payload,
        );
    }

    if matches!(payload, HitKind::Sky) {
        light.color * INTENSITY_SCALE
    } else {
        Vec3::ZERO
    }
}

#[inline(always)]
fn spotlight_penumbra(light: &Light, to_light: Vec3) -> Option<f32> {
    let light_dot = (-to_light).dot(light.direction);

    if light_dot < light.penumbra_end {
        return None;
    }

    if light_dot < light.penumbra_start {
        let range = (light.penumbra_start - light.penumbra_end).max(1e-6);
        let t = ((light_dot - light.penumbra_end) / range).clamp(0.0, 1.0);
        let exponent = light.exponent;
        let scale = if exponent > 1e-6 && (exponent - 1.0).abs() > 1e-6 {
            t.powf(exponent)
        } else {
            t
        };
        Some(scale)
    } else {
        Some(1.0)
    }
}

#[inline(always)]
fn distance_attenuation(light: &Light, dist: f32) -> f32 {
    let attn_dist = if light.radius > 0.0 {
        dist.max(1.0).min(light.radius)
    } else {
        dist.max(1.0)
    };

    let attenuation = light.constant_attn
        + light.linear_attn * attn_dist
        + light.quadratic_attn * attn_dist * attn_dist;

    if attenuation > 1e-6 {
        1.0 / attenuation
    } else {
        1.0
    }
}

#[inline(always)]
pub fn contribute_positional(
    light: &Light,
    sample_pos: Vec3,
    sample_normal: Vec3,
    tlas: &AccelerationStructure,
    payload: &mut RayPayload,
) -> Vec3 {
    let diff = sample_pos - light.position;
    let dist = diff.length();

    if dist <= 1e-6 {
        return Vec3::ZERO;
    }

    let to_light = -diff / dist;
    let cos_angle = sample_normal.dot(to_light);

    if cos_angle <= 0.0 {
        return Vec3::ZERO;
    }

    let n_dot_l = if light.ty == EmitType::QuakeLight {
        1.0
    } else {
        cos_angle
    };

    let mut penumbra_scale = 1.0_f32;
    if light.ty == EmitType::Spotlight {
        match spotlight_penumbra(light, to_light) {
            Some(scale) => penumbra_scale = scale,
            None => return Vec3::ZERO,
        }
    }

    if light.ty == EmitType::Surface {
        if (-to_light).dot(light.direction) < 0.0 {
            return Vec3::ZERO;
        }
    }

    let ray_origin = sample_pos + sample_normal * SHADOW_EPSILON;
    let t_max = (dist - SHADOW_EPSILON * cos_angle).max(0.0);

    *payload = HitKind::Miss;

    unsafe {
        tlas.trace_ray(
            RayFlags::OPAQUE,
            0xFF,
            0,
            0,
            0,
            ray_origin,
            0.001,
            to_light,
            t_max,
            payload,
        );
    }

    if matches!(payload, HitKind::Hit) {
        return Vec3::ZERO;
    }

    let inv_attn = distance_attenuation(light, dist);

    light.color * (INTENSITY_SCALE * penumbra_scale * inv_attn)
}
