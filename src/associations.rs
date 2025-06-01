use std::cell::Ref;

use bsp::Bsp;
use lump_definitions::source::{Edge, Face, LumpDefinition, Plane, TextureInfo};

pub trait Associated<T: ?Sized> {
    fn associated<'a>(&self, bsp: &'a Bsp<'a>) -> Ref<'a, T>;
}

macro_rules! association {
    ($type:ty, $lump:path, $expression:tt -> [$assoc_ty:ty]) => {
        impl Associated<[$assoc_ty]> for $type {
            fn associated<'a>(&self, bsp: &'a bsp::Bsp<'a>) -> Ref<'a, [$assoc_ty]> {
                association!(@inner self, bsp, $lump, $expression -> $assoc_ty)
            }
        }
    };
    ($type:ty, $lump:path, $expression:tt -> $assoc_ty:ty) => {
        impl Associated<$assoc_ty> for $type {
            fn associated<'a>(&self, bsp: &'a bsp::Bsp<'a>) -> Ref<'a, $assoc_ty> {
                association!(@inner self, bsp, $lump, $expression -> $assoc_ty)
            }
        }
    };

    (@inner $self:ident, $bsp:ident, $lump:path, $expression:tt -> $assoc_ty:ty) => {{
        let lump = $bsp
            .lump_cast::<[$assoc_ty], _>($lump)
            .expect("Failed to get lump");

        Ref::map(lump, |lump| &association!(@expression $self, lump, $expression))
    }};

    // Rule for [a]
    (@expression $self:ident, $lump:ident, [$field:ident]) => {
        $lump[$self.$field as usize]
    };

    // Rule for [a..b]
    (@expression $self:ident, $lump:ident, [$field1:ident..$field2:ident]) => {
        $lump[$self.$field1 as usize..$self.$field2 as usize]
    };

    // Rule for [a..+b] (a .. a + b)
    (@expression $self:ident, $lump:ident, [$field1:ident..+$field2:ident]) => {
        $lump[$self.$field1 as usize..$self.$field1 as usize + $self.$field2 as usize]
    };
}

association!(Face, LumpDefinition::TextureInfo, [texture_info_index] -> TextureInfo);
association!(Face, LumpDefinition::Planes, [plane_index] -> Plane);
association!(Face, LumpDefinition::Edges, [edge_index..+edge_count] -> [Edge]);
