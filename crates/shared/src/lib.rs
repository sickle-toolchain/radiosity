//! Data types shared between CPU and GPU

#![no_std]

use core::convert::Into;
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
