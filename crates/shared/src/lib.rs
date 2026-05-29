//! Data types shared between CPU and GPU

#![no_std]

use core::convert::Into;
pub use lump_definitions::source::EmitType;
use spirv_std::glam::Vec3;

#[repr(C, align(16))]
#[derive(Default, Clone, Copy, Debug)]
pub struct AlignedVec3(pub Vec3);

impl From<Vec3> for AlignedVec3 {
    fn from(value: Vec3) -> Self {
        Self(value)
    }
}

impl From<AlignedVec3> for Vec3 {
    fn from(val: AlignedVec3) -> Self {
        val.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TexelData {
    pub position: AlignedVec3,
    pub normal: AlignedVec3,
}

impl TexelData {
    pub fn new(position: Vec3, normal: Vec3) -> Self {
        Self {
            position: position.into(),
            normal: normal.into(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Light {
    pub position: AlignedVec3,
    pub color: AlignedVec3,
    pub direction: AlignedVec3,
    pub ty: EmitType,
    pub radius: f32,
    pub constant_attn: f32,
    pub linear_attn: f32,
    pub quadratic_attn: f32,
    pub penumbra_start: f32,
    pub penumbra_end: f32,
    pub exponent: f32,
}

impl core::fmt::Debug for Light {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Light")
            .field("position", &self.position)
            .field("ty", &self.ty)
            .field("color", &self.color)
            .field("radius", &self.radius)
            .field("direction", &self.direction)
            .field("penumbra_start", &self.penumbra_start)
            .field("constant_attn", &self.constant_attn)
            .field("linear_attn", &self.linear_attn)
            .field("quadratic_attn", &self.quadratic_attn)
            .field("penumbra_end", &self.penumbra_end)
            .field("exponent", &self.exponent)
            .finish()
    }
}

unsafe impl bytemuck::Zeroable for Light {}
unsafe impl bytemuck::Pod for Light {}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Sky {
    pub sun_direction: AlignedVec3,
    pub sun_color: AlignedVec3,
    pub ambient_color: AlignedVec3,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RayPayload {
    pub hit_pos: Vec3,
    pub hit_normal: Vec3,
    pub hit: u32,
}
