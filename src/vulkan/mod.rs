mod buffer;

use anyhow::{Context, Result, bail};
use log::{Level, log};
use std::collections::HashSet;
use std::ffi::{CStr, c_void};

use ash::ext::debug_utils;
use ash::vk::{
    CommandPool, PhysicalDevice, PhysicalDeviceMemoryProperties, PhysicalDeviceProperties, Queue,
};
use ash::{Device, Entry, Instance, vk};

pub use buffer::Buffer;

extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let level = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Level::Debug,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::Error,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Level::Info,
        _ => unreachable!(),
    };

    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) }.to_string_lossy();
    log!(target: "vulkan", level, "{message}");

    vk::FALSE
}

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,

    pub physical_device: PhysicalDevice,
    pub physical_device_properties: PhysicalDeviceProperties,
    pub physical_device_memory_properties: PhysicalDeviceMemoryProperties,

    pub device: Device,
    pub queue: Queue,
    pub queue_family_index: u32,

    pub pool: CommandPool,
}

impl VulkanContext {
    pub fn new(
        instance_layers: &[&CStr],
        device_extensions: &[&CStr],
        device_id: Option<u32>,
    ) -> Result<Self> {
        let entry = if cfg!(feature = "ash-linked") {
            Entry::linked()
        } else {
            unsafe { Entry::load() }?
        };

        let instance_layer_properties = unsafe { entry.enumerate_instance_layer_properties() }?;
        let supported_layers: Vec<&CStr> = instance_layer_properties
            .iter()
            .filter_map(|p| p.layer_name_as_c_str().ok())
            .collect();

        if let Some(layer) = instance_layers
            .iter()
            .find(|l| !supported_layers.contains(*l))
        {
            bail!("Layer '{}' is not supported", layer.to_string_lossy());
        }

        let instance = {
            let application_info = vk::ApplicationInfo::default()
                .application_from_env()
                .api_version(vk::API_VERSION_1_2);

            let enabled_extension_names = vec![debug_utils::NAME.as_ptr()];
            let mut debug_utils_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                )
                .pfn_user_callback(Some(vulkan_debug_utils_callback));

            let enabled_layer_names = instance_layers
                .iter()
                .map(|l| l.as_ptr())
                .collect::<Vec<_>>();

            let instance_create_info = vk::InstanceCreateInfo::default()
                .application_info(&application_info)
                .enabled_layer_names(enabled_layer_names.as_slice())
                .enabled_extension_names(enabled_extension_names.as_slice())
                .push_next(&mut debug_utils_create_info);

            unsafe { entry.create_instance(&instance_create_info, None) }
                .context("Failed to create instance")?
        };

        let physical_device = if let Some(id) = device_id {
            unsafe { instance.enumerate_physical_devices()? }
                .iter()
                .find(|&&physical_device| {
                    let props = unsafe { instance.get_physical_device_properties(physical_device) };
                    props.device_id == id
                })
                .copied()
        } else {
            unsafe { instance.enumerate_physical_devices() }?
                .into_iter()
                .find(|&physical_device| {
                    unsafe { instance.enumerate_device_extension_properties(physical_device) }
                        .map(|exts| {
                            let set: HashSet<&CStr> = exts
                                .iter()
                                .map(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) })
                                .collect();

                            device_extensions.iter().all(|ext| set.contains(ext))
                        })
                        .unwrap_or(false)
                })
        }
        .context("Failed to find physical device")?;

        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let queue_family_index =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                .into_iter()
                .enumerate()
                .find(|(_, device_properties)| {
                    device_properties.queue_count > 0
                        && device_properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS)
                })
                .map(|(idx, _)| idx as u32)
                .context("Failed to find queue family index")?;

        let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0])];

        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut features2);
        };

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .vulkan_memory_model(true);

        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);

        let mut raytracing_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

        let enabled_extension_names = device_extensions
            .iter()
            .map(|c| c.as_ptr())
            .collect::<Vec<_>>();

        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut features2)
            .push_next(&mut features12)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut raytracing_pipeline_features)
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(enabled_extension_names.as_slice());

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .context("Failed to create device")?;

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool = {
            let command_pool_create_info =
                vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);

            unsafe { device.create_command_pool(&command_pool_create_info, None) }
                .context("Failed to create command pool")?
        };

        Ok(Self {
            entry,
            instance,

            physical_device,
            physical_device_properties,
            physical_device_memory_properties,

            device,
            queue,
            queue_family_index,

            pool,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub trait GeometryIndex {
    fn vk_index_type() -> vk::IndexType;
}

impl GeometryIndex for u16 {
    fn vk_index_type() -> vk::IndexType {
        vk::IndexType::UINT16
    }
}

impl GeometryIndex for u32 {
    fn vk_index_type() -> vk::IndexType {
        vk::IndexType::UINT32
    }
}

pub trait ApplicationInfoExt {
    fn application_from_env(self) -> Self;
}

impl ApplicationInfoExt for vk::ApplicationInfo<'_> {
    fn application_from_env(self) -> Self {
        let application_name =
            CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_NAME"), "\0").as_bytes())
                .expect("invalid package name");

        let major = env!("CARGO_PKG_VERSION_MAJOR")
            .parse::<u32>()
            .expect("invalid major version");
        let minor = env!("CARGO_PKG_VERSION_MINOR")
            .parse::<u32>()
            .expect("invalid minor version");
        let patch = env!("CARGO_PKG_VERSION_PATCH")
            .parse::<u32>()
            .expect("invalid patch version");

        self.application_name(application_name)
            .application_version(vk::make_api_version(0, major, minor, patch))
    }
}

pub trait PhysicalDeviceMemoryPropertiesExt {
    fn mem_ty_idx(
        &self,
        required_bits: u32,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32>;
}

impl PhysicalDeviceMemoryPropertiesExt for vk::PhysicalDeviceMemoryProperties {
    fn mem_ty_idx(
        &self,
        required_bits: u32,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for idx in 0..self.memory_type_count {
            let memory_properties = self.memory_types[idx as usize].property_flags;

            if (required_bits & (1 << idx)) == 1
                && (memory_properties & required_properties) == required_properties
            {
                return Some(idx);
            }
        }

        None
    }
}
