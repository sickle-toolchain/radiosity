use std::path::PathBuf;

use bsp::Bsp;
use clap::Parser;
use lump_definitions::source::{ColorRGBExp32, Face, LumpDefinition};

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

    let faces = bsp
        .lump_cast::<[Face], _>(if args.hdr {
            LumpDefinition::FacesHdr
        } else {
            LumpDefinition::Faces
        })
        .expect("failed to parse [Face]");

    dbg!(faces.len());

    let lighting = bsp
        .lump_cast::<[ColorRGBExp32], _>(if args.hdr {
            LumpDefinition::LightingHdr
        } else {
            LumpDefinition::Lighting
        })
        .expect("failed to parse [ColorRGBExp32]");

    dbg!(lighting.len());

    let world_lights = bsp
        .lump_cast::<[ColorRGBExp32], _>(if args.hdr {
            LumpDefinition::WorldLightsHdr
        } else {
            LumpDefinition::WorldLights
        })
        .expect("failed to parse [WorldLight]");

    dbg!(world_lights.len());

    Ok(())
}
