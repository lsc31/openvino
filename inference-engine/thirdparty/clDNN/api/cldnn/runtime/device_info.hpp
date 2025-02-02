// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_device GPU Device
/// @{

/// @brief Enumeration of supported device types
enum class device_type {
    integrated_gpu = 0,
    discrete_gpu = 1
};

/// @brief Defines version of GFX IP
struct gfx_version {
    uint16_t major;
    uint8_t minor;
    uint8_t revision;
};

/// @brief Information about the device properties and capabilities.
struct device_info {
    uint32_t execution_units_count;             ///< Number of available execution units.
    uint32_t gpu_frequency;                     ///< Clock frequency in MHz.
    uint32_t max_threads_per_execution_unit;    ///< Number of available HW threads on EU.
    uint32_t max_threads_per_device;            ///< Maximum number of HW threads on device.

    uint64_t max_work_group_size;               ///< Maximum number of work-items in a work-group executing a kernel using the data parallel execution model.
    uint64_t max_local_mem_size;                ///< Maximum size of local memory arena in bytes.
    uint64_t max_global_mem_size;               ///< Maximum size of global device memory in bytes.
    uint64_t max_alloc_mem_size;                ///< Maximum size of memory object allocation in bytes.

    uint64_t max_image2d_width;                 ///< Maximum image 2d width supported by the device.
    uint64_t max_image2d_height;                ///< Maximum image 2d height supported by the device.

    bool supports_fp16;                         ///< Does engine support FP16.
    bool supports_fp64;                         ///< Does engine support FP64.
    bool supports_fp16_denorms;                 ///< Does engine support denormalized FP16.
    bool supports_subgroups;                    ///< Does engine support cl_intel_subgroups extension.
    bool supports_subgroups_short;              ///< Does engine support cl_intel_subgroups_short extension.
    bool supports_subgroups_char;               ///< Does engine support cl_intel_subgroups_char extension.
    bool supports_local_block_io;               ///< Does engine support cl_intel_subgroup_local_block_io extension.
    bool supports_queue_families;               ///< Does engine support cl_intel_command_queue_families extension.
    bool supports_image;                        ///< Does engine support images (CL_DEVICE_IMAGE_SUPPORT cap).

    bool supports_imad;                         ///< Does engine support int8 mad.
    bool supports_immad;                        ///< Does engine support int8 multi mad.

    bool supports_usm;                          ///< Does engine support unified shared memory.

    uint32_t vendor_id;                         ///< Vendor ID
    std::string dev_name;                       ///< Device ID string
    std::string driver_version;                 ///< Version of OpenCL driver

    device_type dev_type;                       ///< Defines type of current GPU device (integrated or discrete)

    gfx_version gfx_ver;                        ///< Defines GFX IP version
    uint32_t device_id;                         ///< ID of current GPU
    uint32_t num_slices;                        ///< Number of slices
    uint32_t num_sub_slices_per_slice;          ///< Number of subslices in a slice
    uint32_t num_eus_per_sub_slice;             ///< Number of execution units per subslice
    uint32_t num_threads_per_eu;                ///< Number of hardware threads per execution unit
};

/// @}

/// @}

}  // namespace cldnn
