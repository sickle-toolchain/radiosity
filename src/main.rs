use std::path::PathBuf;

use bsp::Bsp;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Path to BSP file
    bsp_path: PathBuf,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let contents = std::fs::read(args.bsp_path)?;
    let bsp = Bsp::parse(&contents).expect("failed to parse BSP file");

    dbg!(bsp);

    Ok(())
}
