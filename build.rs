use std::error::Error;
use std::path::Path;

use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    let shader_crate_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("radiosity-shader");
    println!(
        "cargo::rerun-if-changed={}",
        shader_crate_path.join("Cargo.toml").display()
    );
    println!(
        "cargo::rerun-if-changed={}",
        shader_crate_path.join("src").join("lib.rs").display()
    );

    SpirvBuilder::new(shader_crate_path, "spirv-unknown-vulkan1.2")
        .extension("SPV_KHR_ray_tracing")
        .capability(Capability::RayTracingKHR)
        .print_metadata(MetadataPrintout::Full)
        .build()?;

    Ok(())
}
