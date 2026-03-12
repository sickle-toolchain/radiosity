use std::cell::Ref;

use bsp::Bsp;
use lump_definitions::source::{Edge, Face, LumpDefinition, Plane, SurfaceEdge, TextureInfo};

/// Trait for types that have associated data in a specific BSP lump.
pub trait Associated<T: ?Sized> {
    fn associated<'a>(&self, bsp: &'a Bsp<'a>) -> Ref<'a, T>;
}

macro_rules! lump_ref {
    ($bsp:expr, $lump:path, $elem:ty, |$l:ident| $body:expr) => {{
        let lump = $bsp
            .lump_cast::<[$elem], _>($lump)
            .expect("Failed to get lump");
        Ref::map(lump, |$l| $body)
    }};
}

macro_rules! association {
    // Scalar: single field index
    ($type:ty, $lump:path, $assoc_ty:ty, [$field:ident]) => {
        impl Associated<$assoc_ty> for $type {
            fn associated<'a>(&self, bsp: &'a Bsp<'a>) -> Ref<'a, $assoc_ty> {
                lump_ref!(bsp, $lump, $assoc_ty, |l| &l[self.$field as usize])
            }
        }
    };
    // Scalar: absolute value of field index
    ($type:ty, $lump:path, $assoc_ty:ty, [|$field:ident|]) => {
        impl Associated<$assoc_ty> for $type {
            fn associated<'a>(&self, bsp: &'a Bsp<'a>) -> Ref<'a, $assoc_ty> {
                lump_ref!(bsp, $lump, $assoc_ty, |l| &l
                    [self.$field.unsigned_abs() as usize])
            }
        }
    };
    // Slice: range [start..end]
    ($type:ty, $lump:path, $assoc_ty:ty, [$start:ident..$end:ident]) => {
        impl Associated<[$assoc_ty]> for $type {
            fn associated<'a>(&self, bsp: &'a Bsp<'a>) -> Ref<'a, [$assoc_ty]> {
                lump_ref!(bsp, $lump, $assoc_ty, |l| &l
                    [self.$start as usize..self.$end as usize])
            }
        }
    };
    // Slice: range [start..+count] (start .. start + count)
    ($type:ty, $lump:path, $assoc_ty:ty, [$start:ident..+$count:ident]) => {
        impl Associated<[$assoc_ty]> for $type {
            fn associated<'a>(&self, bsp: &'a Bsp<'a>) -> Ref<'a, [$assoc_ty]> {
                lump_ref!(bsp, $lump, $assoc_ty, |l| &l[self.$start as usize
                    ..self.$start as usize + self.$count as usize])
            }
        }
    };
}

association!(
    Face,
    LumpDefinition::TextureInfo,
    TextureInfo,
    [texture_info_index]
);
association!(Face, LumpDefinition::Planes, Plane, [plane_index]);
association!(Face, LumpDefinition::SurfaceEdges, SurfaceEdge, [edge_index..+edge_count]);
association!(SurfaceEdge, LumpDefinition::Edges, Edge, [|edge_index|]);
