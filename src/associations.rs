use std::cell::Ref;

use bsp::Bsp;
use lump_definitions::source::{Face, LumpDefinition, Plane, TextureInfo};


pub trait Associated<T: ?Sized> {
    fn associated<'a>(&self, bsp: &'a Bsp<'a>) -> Ref<'a, T>;
}

impl Associated<TextureInfo> for Face {
    fn associated<'a>(&self, bsp: &'a Bsp<'a>) -> Ref<'a, TextureInfo> {
        let texture_info = bsp
            .lump_cast::<[TextureInfo], _>(LumpDefinition::TextureInfo)
            .expect("Failed to get LumpDefinition::TextureInfo");

        assert!((self.texture_info_index as usize) < texture_info.len());
        Ref::map(texture_info, |texture_info| &texture_info[self.texture_info_index as usize])
    }
}

impl Associated<Plane> for Face {
    fn associated<'a>(&self, bsp: &'a Bsp<'a>) -> Ref<'a, Plane> {
        let planes = bsp
            .lump_cast::<[Plane], _>(LumpDefinition::Planes)
            .expect("Failed to get LumpDefinition::Planes");

        assert!((self.plane_index as usize) < planes.len());
        Ref::map(planes, |planes| &planes[self.plane_index as usize])
    }
}
