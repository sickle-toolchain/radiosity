use std::error::Error;
use std::path::Path;

use spirv_builder::{Capability, SpirvBuilder};

fn main() -> Result<(), Box<dyn Error>> {
    let shader_crate_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("shader");

    unsafe {
        std::env::set_var(
            "RUSTGPU_RUSTFLAGS",
            "-Ctarget-feature=+RayTracingPositionFetchKHR",
        );
    }

    let mut builder = SpirvBuilder::new(shader_crate_path, "spirv-unknown-vulkan1.3")
        .extension("SPV_KHR_ray_tracing")
        .extension("SPV_KHR_ray_tracing_position_fetch")
        .capability(Capability::RayTracingKHR)
        .preserve_bindings(true);

    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(true);

    builder.build()?;
    Ok(())
}
