use std::{borrow::Cow, path::PathBuf};

use bsp::Bsp;
use clap::Parser;
use lump_definitions::source::{
    ColorRGBExp32, Face, LumpDefinition, Plane, TextureData, TextureInfo, WorldLight,
    LIGHTMAP_COUNT,
};
use zerocopy::IntoBytes;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    /// Path to BSP file
    bsp_path: PathBuf,

    /// Use high dynamic range lumps
    #[arg(long)]
    hdr: bool,
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

    let planes = bsp
        .lump_cast::<[Plane], _>(LumpDefinition::Planes)
        .expect("Failed to get LumpDefinition::Planes");

    let texture_info = bsp
        .lump_cast::<[TextureInfo], _>(LumpDefinition::TextureInfo)
        .expect("Failed to get LumpDefinition::TextureInfo");

    let texture_data = bsp
        .lump_cast::<[TextureData], _>(LumpDefinition::TextureData)
        .expect("Failed to get LumpDefinition::TextureInfo");

    let (_, mut lighting) = bsp.lump_mut(if args.hdr {
        LumpDefinition::LightingHdr
    } else {
        LumpDefinition::Lighting
    });

    let lightmaps = faces
        .iter_mut()
        .map(|face| {
            let texture_info = &texture_info[face.texture_info_index as usize];
            let texture_data = &texture_data[texture_info.texture_data as usize];

            let cross = [
                texture_info.luxels[1][1] * texture_info.luxels[0][2]
                    - texture_info.luxels[1][2] * texture_info.luxels[0][1],
                texture_info.luxels[1][2] * texture_info.luxels[0][0]
                    - texture_info.luxels[1][0] * texture_info.luxels[0][2],
                texture_info.luxels[1][0] * texture_info.luxels[0][1]
                    - texture_info.luxels[1][1] * texture_info.luxels[0][0],
            ];

            let plane = &planes[face.plane_index as usize];
            let det = -(plane
                .normal
                .iter()
                .zip(cross.iter())
                .map(|(x, y)| x * y)
                .sum::<f32>());
            assert!(det.abs() >= 1.0e-20, "face vectors parallel to face normal");

            let luxel_to_world = [
                [
                    (plane.normal[2] * texture_info.luxels[1][1]
                        - plane.normal[1] * texture_info.luxels[1][2])
                        / det,
                    (plane.normal[0] * texture_info.luxels[1][2]
                        - plane.normal[2] * texture_info.luxels[1][0])
                        / det,
                    (plane.normal[1] * texture_info.luxels[1][0]
                        - plane.normal[0] * texture_info.luxels[1][1])
                        / det,
                ],
                [
                    (plane.normal[1] * texture_info.luxels[0][2]
                        - plane.normal[2] * texture_info.luxels[0][1])
                        / det,
                    (plane.normal[2] * texture_info.luxels[0][0]
                        - plane.normal[0] * texture_info.luxels[0][2])
                        / det,
                    (plane.normal[0] * texture_info.luxels[0][1]
                        - plane.normal[1] * texture_info.luxels[0][0])
                        / det,
                ],
            ];

            let luxel_origin: [f32; 3] = std::array::from_fn(|index| {
                -(plane.dist * cross[index]) / det
                    + luxel_to_world[0][index] * -texture_info.luxels[0][3]
                    + luxel_to_world[1][index] * -texture_info.luxels[1][3]
            });

            let width = (face.lightmap.maxs[0] + 1) as usize;
            let height = (face.lightmap.maxs[1] + 1) as usize;
            let luxel_count = width * height;

            let lightmap = world_lights.iter().fold(
                vec![
                    ColorRGBExp32 {
                        r: (255.0 * texture_data.reflectivity[0]) as u8,
                        g: (255.0 * texture_data.reflectivity[1]) as u8,
                        b: (255.0 * texture_data.reflectivity[2]) as u8,
                        exponent: 0
                    };
                    luxel_count * LIGHTMAP_COUNT
                ],
                |mut lightmap, light| {
                    for style in 0..LIGHTMAP_COUNT {
                        let style_offset = luxel_count * style;
                        let luxels = &mut lightmap[style_offset..style_offset + luxel_count];

                        (0..height).for_each(|t| {
                            (0..width).for_each(|s| {
                                let luxel_position: [f32; 3] = std::array::from_fn(|index| {
                                    luxel_origin[index]
                                        + luxel_to_world[0][index]
                                            * (s as i32 + face.lightmap.mins[0]) as f32
                                        + luxel_to_world[1][index]
                                            * (t as i32 + face.lightmap.mins[1]) as f32
                                });

                                // Check if luxel position is within 128 units of the light origin
                                if ((light.origin[0] - luxel_position[0]).powi(2)
                                    + (light.origin[1] - luxel_position[1]).powi(2)
                                    + (light.origin[2] - luxel_position[2]).powi(2))
                                .sqrt()
                                    < 256.0
                                {
                                    luxels[s + t * width].exponent = 1;
                                }
                            });
                        });
                    }

                    lightmap
                },
            );

            (face, lightmap)
        })
        .fold(Vec::new(), |mut acc, (face, lightmap)| {
            // Average colors for face
            acc.extend_from_slice(
                &[ColorRGBExp32 {
                    r: 255,
                    g: 255,
                    b: 255,
                    exponent: 0,
                }; LIGHTMAP_COUNT],
            );

            face.styles = [0; LIGHTMAP_COUNT];
            face.light_offset = (acc.len() * size_of::<ColorRGBExp32>()) as i32;

            acc.extend_from_slice(&lightmap);

            acc
        });

    *lighting = Cow::Owned(lightmaps.as_bytes().to_owned());

    // We must drop anything we have mutable access to so
    // write_to_io can gain immutable access.
    drop(faces);
    drop(lighting);

    bsp.write_to_io(&mut std::fs::File::create("out.bsp").unwrap())
        .expect("writing to io failed");

    Ok(())
}
