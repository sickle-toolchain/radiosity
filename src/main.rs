use std::{borrow::Cow, path::PathBuf};

use bsp::Bsp;
use clap::Parser;
use glam::Vec3;
use lump_definitions::source::{
    ColorRGBExp32, Face, Lightmap, LumpDefinition, Plane, TextureInfo,
    WorldLight,
};
use zerocopy::IntoBytes;

use radiosity::Associated;

#[derive(Parser, Debug)]
struct Args {
    /// Path to BSP file
    bsp_path: PathBuf,

    /// Use high dynamic range lumps
    #[arg(long)]
    hdr: bool,
}

pub struct LuxelMapping {
    lightmap: Lightmap,
    luxel_origin: Vec3,
    luxel_to_worldspace: [Vec3; 2],
    world_to_luxelspace: [Vec3; 2],
}

impl LuxelMapping {
    pub fn new<'a>(bsp: &'a Bsp<'a>, face: &Face) -> Self {
        let plane = <Face as Associated<Plane>>::associated(face, bsp);
        let texture_info = <Face as Associated<TextureInfo>>::associated(face, bsp);
        let world_to_luxelspace = texture_info.luxels.map(|s| Vec3::from_array(s.xyz));

        let s_luxels = Vec3::from_array(texture_info.luxels[0].xyz);
        let t_luxels = Vec3::from_array(texture_info.luxels[1].xyz);

        let cross = t_luxels.cross(s_luxels);

        let normal = Vec3::from_array(plane.normal);
        let det = -normal.dot(cross);
        assert!(det.abs() >= 1.0e-20, "face vectors parallel to face normal");

        let luxel_to_worldspace = [t_luxels.cross(normal) / det, normal.cross(s_luxels) / det];

        let luxel_origin = -(plane.dist * cross) / det
            + luxel_to_worldspace[0] * -texture_info.luxels[0].offset
            + luxel_to_worldspace[1] * -texture_info.luxels[1].offset;

        Self {
            lightmap: face.lightmap,
            luxel_origin,
            luxel_to_worldspace,
            world_to_luxelspace,
        }
    }

    /// Converts luxel space coordinates to world space coordinates.
    ///
    /// # Arguments
    ///
    /// * `s` - The s-coordinate in luxel space.
    /// * `t` - The t-coordinate in luxel space.
    ///
    /// # Returns
    ///
    /// * `Vec3` - The corresponding world space coordinates.
    pub fn luxel_to_world(&self, s: f32, t: f32) -> Vec3 {
        let [s_min, t_min] = self.lightmap.mins;
        let (s, t) = (s + s_min as f32, t + t_min as f32);

        self.luxel_origin + self.luxel_to_worldspace[0] * s + self.luxel_to_worldspace[1] * t
    }

    /// Converts world space coordinates to luxel coordinates.
    ///
    /// # Arguments
    ///
    /// * `world` - The world space coordinates.
    ///
    /// # Returns
    ///
    /// * `(f32, f32)` - The corresponding luxel space coordinates (s, t).
    pub fn world_to_luxel(&self, mut world: Vec3) -> (f32, f32) {
        world -= self.luxel_origin;
        let s = world.dot(self.world_to_luxelspace[0]) - self.lightmap.mins[0] as f32;
        let t = world.dot(self.world_to_luxelspace[1]) - self.lightmap.mins[1] as f32;

        (s, t)
    }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let contents = std::fs::read(args.bsp_path)?;
    let bsp = Bsp::parse(&contents).expect("failed to parse BSP file");

    let mut faces = bsp
        .lump_cast_mut::<[Face], _>(if args.hdr {
            LumpDefinition::FacesHdr
        } else {
            LumpDefinition::Faces
        })
        .expect("Failed to get LumpDefinition::Faces");

    let world_lights = bsp
        .lump_cast::<[WorldLight], _>(if args.hdr {
            LumpDefinition::WorldLightsHdr
        } else {
            LumpDefinition::WorldLights
        })
        .expect("Failed to get LumpDefinition::WorldLights");

    let mut lighting = bsp.lump_mut(if args.hdr {
        LumpDefinition::LightingHdr
    } else {
        LumpDefinition::Lighting
    });

    let lightmaps = faces
        .iter_mut()
        .map(|face| (LuxelMapping::new(&bsp, face), face))
        .fold(vec![], |mut acc, (mapping, face)| {
            let width = (face.lightmap.maxs[0] + 1) as usize;
            let height = (face.lightmap.maxs[1] + 1) as usize;

            // Accumolate lighting for luxels
            let lightmap = world_lights.iter().fold(
                vec![ColorRGBExp32::default(); width * height],
                |mut acc, light| {
                    (0..height).for_each(|t| {
                        (0..width).for_each(|s| {
                            let light_origin = Vec3::from_array(light.origin);
                            let position = mapping.luxel_to_world(s as f32, t as f32);

                            let distance = light_origin.distance(position) as u8;

                            // Check if luxel position is within 255 units of the light origin
                            if distance < u8::MAX {
                                let color = u8::MAX.saturating_sub(distance);
                                let luxel = &mut acc[s + t * width];

                                luxel.r = color;
                                luxel.g = color;
                                luxel.b = color;
                                luxel.exponent = 0;
                            }
                        });
                    });

                    acc
                },
            );

            face.styles = [0, 255, 255, 255];
            face.light_offset = (acc.len() * size_of::<ColorRGBExp32>()) as i32;

            acc.extend_from_slice(&lightmap);
            acc
        });

    // Replace lighting lump
    lighting.data = Cow::Owned(lightmaps.as_bytes().to_owned());

    // We must drop anything we have mutable access to so
    // write_to_io can gain immutable access.
    drop(faces);
    drop(lighting);

    bsp.write_to_io(&mut std::fs::File::create("out.bsp").unwrap())
        .expect("writing to io failed");

    Ok(())
}
