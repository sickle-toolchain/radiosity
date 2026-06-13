use std::error::Error;
use std::path::Path;

use spirv_builder::{Capability, SpirvBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    let shader_crate_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("shader");

    let mut builder = SpirvBuilder::new(shader_crate_path, "spirv-unknown-vulkan1.3")
        .extension("SPV_KHR_ray_query")
        .capability(Capability::RayQueryKHR)
        .preserve_bindings(true);

    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(true);

    builder.build()?;
    Ok(())
}
