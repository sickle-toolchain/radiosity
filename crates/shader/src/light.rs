use spirv_std::glam::{Vec2, Vec3};
use spirv_std::ray_tracing::{AccelerationStructure, RayFlags};

#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use core::f32::consts::PI;

use shared::{EmitType, Light, MASK_ALL, MASK_SOLID, Sky, TexelData};

use crate::{HitKind, RayPayload};

pub const SHADOW_EPSILON: f32 = 0.25;

const RAY_TMIN: f32 = 0.001;
const RAY_TMAX: f32 = 1.0e6;

const AMBIENT_SAMPLES: u32 = 16;

fn radical_inverse_base2(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new(i as f32 / n as f32, radical_inverse_base2(i))
}

pub fn jittered_position(texel: &TexelData, sample_index: u32, samples_per_luxel: u32) -> Vec3 {
    let j = hammersley(sample_index, samples_per_luxel);
    texel.position.0 + texel.tangent.0 * (j.x - 0.5) + texel.bitangent.0 * (j.y - 0.5)
}

fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        Vec3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

fn sample_uniform_hemisphere(normal: Vec3, u: Vec2) -> Vec3 {
    let cos_theta = u.x;
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u.y;
    let (t, b) = orthonormal_basis(normal);
    (t * (sin_theta * phi.cos()) + b * (sin_theta * phi.sin()) + normal * cos_theta).normalize()
}

fn sample_cone(axis: Vec3, sin_radius: f32, u: Vec2) -> Vec3 {
    if sin_radius <= 0.0 {
        return axis;
    }
    let cos_max = (1.0 - sin_radius * sin_radius).max(0.0).sqrt();
    let cos_theta = 1.0 - u.x * (1.0 - cos_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u.y;
    let (t, b) = orthonormal_basis(axis);
    (t * (sin_theta * phi.cos()) + b * (sin_theta * phi.sin()) + axis * cos_theta).normalize()
}

#[inline(always)]
pub fn contribute_sky(
    sky: &Sky,
    sample_pos: Vec3,
    sample_normal: Vec3,
    tlas: &AccelerationStructure,
    payload: &mut RayPayload,
    sample_index: u32,
    samples_per_luxel: u32,
) -> Vec3 {
    let ray_origin = sample_pos + sample_normal * SHADOW_EPSILON;
    let mut result = Vec3::ZERO;

    let ambient = sky.ambient_color.0;
    if ambient.length_squared() > 0.0 {
        let total_ambient = samples_per_luxel * AMBIENT_SAMPLES;
        let mut weighted_visible = 0.0_f32;
        let mut total_weight = 0.0_f32;
        for a in 0..AMBIENT_SAMPLES {
            let gi = a * samples_per_luxel + sample_index;
            let u = hammersley(gi, total_ambient);
            let dir = sample_uniform_hemisphere(sample_normal, u);
            let cos_w = sample_normal.dot(dir);
            if cos_w <= 0.0 {
                continue;
            }
            total_weight += cos_w;

            *payload = HitKind::Miss;
            unsafe {
                tlas.trace_ray(
                    RayFlags::OPAQUE,
                    MASK_ALL as i32,
                    0,
                    0,
                    0,
                    ray_origin,
                    RAY_TMIN,
                    dir,
                    RAY_TMAX,
                    payload,
                );
            }
            if !matches!(payload, HitKind::Hit) {
                weighted_visible += cos_w;
            }
        }
        if total_weight > 0.0 {
            result += ambient * (weighted_visible / total_weight);
        }
    }

    let sun_color = sky.sun_color.0;
    if sun_color.length_squared() > 0.0 {
        let to_sun = -sky.sun_direction.0;
        let n_dot_l = sample_normal.dot(to_sun);
        if n_dot_l > 0.0 {
            let u = hammersley(sample_index, samples_per_luxel);
            let ray_dir = sample_cone(to_sun, sky.sun_spread, u);

            *payload = HitKind::Miss;
            unsafe {
                tlas.trace_ray(
                    RayFlags::OPAQUE,
                    MASK_ALL as i32,
                    0,
                    0,
                    0,
                    ray_origin,
                    RAY_TMIN,
                    ray_dir,
                    RAY_TMAX,
                    payload,
                );
            }

            if !matches!(payload, HitKind::Hit) {
                result += sun_color * n_dot_l;
            }
        }
    }

    result
}

#[inline(always)]
fn spotlight_penumbra(light: &Light, to_light: Vec3) -> Option<f32> {
    let light_dot = (-to_light).dot(light.direction.0);

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
    let diff = sample_pos - light.position.0;
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

    let mut angular = 1.0_f32;
    match light.ty {
        EmitType::Spotlight => {
            let axis_dot = (-to_light).dot(light.direction.0);
            match spotlight_penumbra(light, to_light) {
                Some(fringe) => angular = fringe * axis_dot,
                None => return Vec3::ZERO,
            }
        }
        EmitType::Surface => {
            let emitter_dot = (-to_light).dot(light.direction.0);
            if emitter_dot <= 0.0 {
                return Vec3::ZERO;
            }
            angular = emitter_dot;
        }
        _ => {}
    }

    let falloff = if light.ty == EmitType::Surface {
        if light.radius > 0.0 && dist > light.radius {
            return Vec3::ZERO;
        }
        1.0 / (dist * dist).max(1.0)
    } else {
        distance_attenuation(light, dist)
    };

    let ray_origin = sample_pos + sample_normal * SHADOW_EPSILON;
    let t_max = (dist - SHADOW_EPSILON * cos_angle).max(0.0);

    *payload = HitKind::Hit;
    unsafe {
        tlas.trace_ray(
            RayFlags::OPAQUE | RayFlags::TERMINATE_ON_FIRST_HIT,
            MASK_SOLID as i32,
            0,
            0,
            0,
            ray_origin,
            RAY_TMIN,
            to_light,
            t_max,
            payload,
        );
    }

    if matches!(payload, HitKind::Hit) {
        return Vec3::ZERO;
    }

    light.color.0 * (n_dot_l * angular * falloff)
}
