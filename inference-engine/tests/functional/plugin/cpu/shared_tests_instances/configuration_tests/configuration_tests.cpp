// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "configuration_tests/configuration_tests.hpp"
#include "threading/ie_istreams_executor.hpp"
#include "ie_plugin_config.hpp"
#if (defined(__APPLE__) || defined(_WIN32))
#include "ie_system_conf.h"
#endif

namespace {
#if (defined(__APPLE__) || defined(_WIN32))
auto defaultBindThreadParameter = InferenceEngine::Parameter{[] {
    auto numaNodes = InferenceEngine::getAvailableNUMANodes();
    if (numaNodes.size() > 1) {
        return std::string{CONFIG_VALUE(NUMA)};
    } else {
        return std::string{CONFIG_VALUE(NO)};
    }
}()};
#else
auto defaultBindThreadParameter = InferenceEngine::Parameter{std::string{CONFIG_VALUE(YES)}};
#endif
INSTANTIATE_TEST_SUITE_P(
    smoke_Basic,
    DefaultConfigurationTest,
    ::testing::Combine(
        ::testing::Values("CPU"),
        ::testing::Values(DefaultParameter{CONFIG_KEY(CPU_BIND_THREAD), defaultBindThreadParameter})),
    DefaultConfigurationTest::getTestCaseName);

}  // namespace