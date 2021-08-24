
//
// Auto generated file by CMake macros cross_compiled_file()
// !! do not modify it !!!
//
#include "/mnt/hdd3/nchennub/newR/vendor/intel/external/project-celadon/mani/openvino/inference-engine/src/mkldnn_plugin/nodes/proposal_imp.hpp"
#include "ie_system_conf.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace AVX2 {
    void proposal_exec(const float* input0, const float* input1,
        std::vector<size_t> dims0, std::array<float, 4> img_info,
        const float* anchors, int* roi_indices,
        float* output0, float* output1, proposal_conf &conf); 
}
namespace ANY {
    void proposal_exec(const float* input0, const float* input1,
        std::vector<size_t> dims0, std::array<float, 4> img_info,
        const float* anchors, int* roi_indices,
        float* output0, float* output1, proposal_conf &conf); 
}
namespace XARCH {

void proposal_exec(const float* input0, const float* input1,
        std::vector<size_t> dims0, std::array<float, 4> img_info,
        const float* anchors, int* roi_indices,
        float* output0, float* output1, proposal_conf &conf) {
#ifndef __clang__
    if (with_cpu_x86_avx2()) {
        return AVX2::proposal_exec(input0, input1, dims0, img_info, anchors, roi_indices, output0, output1, conf);
    }
#endif
    if (true) {
        return ANY::proposal_exec(input0, input1, dims0, img_info, anchors, roi_indices, output0, output1, conf);
    }
}

}
}  // namespace InferenceEngine
}  // namespace Extensions
}  // namespace Cpu
