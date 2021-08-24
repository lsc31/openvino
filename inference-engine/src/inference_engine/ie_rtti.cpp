// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include <map>

#include <details/ie_exception.hpp>
#include <ie_blob.h>
#include <ie_parameter.hpp>
#include <ie_iextension.h>
#include <ie_extension.h>
#include <ie_preprocess.hpp>
#include <inference_engine.hpp>
#include <ie_precision.hpp>
#include <exec_graph_info.hpp>

#include <ngraph/opsets/opset.hpp>
#if defined(ENABLE_GNA)
#include <gna-api-types-xnn.h>
#endif

using namespace InferenceEngine;

#ifdef __clang__

template <typename T,
          typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type,
          typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type>
    bool Blob::is() noexcept {
        return dynamic_cast<T*>(this) != nullptr;
    }

template <typename T,
          typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type,
          typename std::enable_if<std::is_base_of<Blob, T>::value, int>::type>
    bool Blob::is() const noexcept {
        return dynamic_cast<const T*>(this) != nullptr;
    }
template bool InferenceEngine::Blob::is<InferenceEngine::MemoryBlob, 0, 0>();
template bool InferenceEngine::Blob::is<InferenceEngine::CompoundBlob, 0, 0>();
template bool InferenceEngine::Blob::is<InferenceEngine::RemoteBlob, 0, 0>();

template <class T>
inline bool Parameter::is() const {
    return empty() ? false : ptr->is(typeid(T));
}

template <class T>
inline bool Parameter::RealData<T>::is(const std::type_info& id) const {
    return id == typeid(T);
}

template <class T>
inline bool Parameter::RealData<T>::operator==(const Parameter::Any& rhs) const {
    return rhs.is(typeid(T)) && equal<T>(*this, rhs);
}


template struct InferenceEngine::Parameter::RealData<int>;
template struct InferenceEngine::Parameter::RealData<bool>;
template struct InferenceEngine::Parameter::RealData<float>;
template struct InferenceEngine::Parameter::RealData<double>;
template struct InferenceEngine::Parameter::RealData<uint32_t>;
template struct InferenceEngine::Parameter::RealData<std::string>;
template struct InferenceEngine::Parameter::RealData<unsigned long>;
template struct InferenceEngine::Parameter::RealData<std::vector<int>>;
template struct InferenceEngine::Parameter::RealData<std::vector<std::string>>;
template struct InferenceEngine::Parameter::RealData<std::vector<unsigned long>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int>>;
template struct InferenceEngine::Parameter::RealData<std::tuple<unsigned int, unsigned int, unsigned int>>;
template struct InferenceEngine::Parameter::RealData<InferenceEngine::Blob::Ptr>;
template bool InferenceEngine::Parameter::is<std::shared_ptr<InferenceEngine::Blob> >() const;
template bool InferenceEngine::Parameter::is<std::shared_ptr<InferenceEngine::Blob const> >() const;
template bool InferenceEngine::Parameter::is<std::basic_string<char, std::char_traits<char>, std::allocator<char> > >() const;
template bool InferenceEngine::Parameter::is<int>() const;
template bool InferenceEngine::Parameter::is<unsigned int>() const;
template bool InferenceEngine::Parameter::is<float>() const;
template bool InferenceEngine::Parameter::is<bool>() const;
template bool InferenceEngine::Parameter::is<std::vector<std::basic_string<char, std::char_traits<char>, std::allocator<char> >,
                             std::allocator<std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >() const;
template bool InferenceEngine::Parameter::is<std::vector<int, std::allocator<int> > >() const;
template bool InferenceEngine::Parameter::is<std::vector<float, std::allocator<float> > >() const;
template bool InferenceEngine::Parameter::is<std::vector<unsigned int, std::allocator<unsigned int> > >() const;
template bool InferenceEngine::Parameter::is<std::tuple<unsigned int, unsigned int, unsigned int> >() const;
template bool InferenceEngine::Parameter::is<std::tuple<unsigned int, unsigned int> >() const;
template bool InferenceEngine::Parameter::is<InferenceEngine::PreProcessInfo>() const;
template bool InferenceEngine::Parameter::is<unsigned long>() const;
template bool InferenceEngine::Parameter::is<std::vector<unsigned long, std::allocator<unsigned long> > >() const;
template bool InferenceEngine::Parameter::is<std::vector<bool, std::allocator<bool> > >() const;

template bool InferenceEngine::Parameter::RealData<std::shared_ptr<InferenceEngine::Blob const> >::is(std::type_info const&) const;
template bool InferenceEngine::Parameter::RealData<InferenceEngine::PreProcessInfo>::is(std::type_info const&)  const;

template bool InferenceEngine::Parameter::RealData<std::vector<unsigned int, std::allocator<unsigned int> > >::is(std::type_info const&) const;
template bool InferenceEngine::Parameter::RealData<std::vector<float, std::allocator<float> > >::is(std::type_info const&) const;
template bool InferenceEngine::Parameter::RealData<std::__bit_reference<std::vector<bool, std::allocator<bool> >, true> >::is(std::type_info const&) const;

template bool InferenceEngine::Parameter::RealData<std::vector<float, std::allocator<float> > >::operator==(InferenceEngine::Parameter::Any const&) const;
template bool InferenceEngine::Parameter::RealData<std::vector<unsigned int, std::allocator<unsigned int> > >::operator==(InferenceEngine::Parameter::Any const&) const;
template bool InferenceEngine::Parameter::RealData<std::shared_ptr<InferenceEngine::Blob const> >::operator==(InferenceEngine::Parameter::Any const&)  const;
template bool InferenceEngine::Parameter::RealData<InferenceEngine::PreProcessInfo>::operator==(InferenceEngine::Parameter::Any const&)  const;
template bool InferenceEngine::Parameter::RealData<std::__bit_reference<std::vector<bool, std::allocator<bool> >, true> >::operator==(InferenceEngine::Parameter::Any const&) const;
#endif  // __clang__
//
// ie_blob.h
//

#ifdef __clang__
template <typename T, typename U>
TBlob<T, U>::~TBlob() {
    free();
}

template class InferenceEngine::TBlob<float>;
template class InferenceEngine::TBlob<double>;
template class InferenceEngine::TBlob<int8_t>;
template class InferenceEngine::TBlob<uint8_t>;
template class InferenceEngine::TBlob<int16_t>;
template class InferenceEngine::TBlob<uint16_t>;
template class InferenceEngine::TBlob<int32_t>;
template class InferenceEngine::TBlob<uint32_t>;
template class InferenceEngine::TBlob<long>;
template class InferenceEngine::TBlob<long long>;
template class InferenceEngine::TBlob<unsigned long>;
template class InferenceEngine::TBlob<unsigned long long>;
#endif  // __clang__
#ifdef __clang__
template <class T>
Precision Precision::fromType(const char* typeName) {
    return Precision(8 * sizeof(T), typeName == nullptr ? typeid(T).name() : typeName);
}

/** @brief checks whether given storage class T can be used to store objects of current precision */
template <class T>
bool Precision::hasStorageType(const char* typeName) const noexcept {
    try {
        if (precisionInfo.value != BIN) {
            if (sizeof(T) != size()) {
                return false;
            }
        }
#define CASE(x, y) \
    case x:        \
        return std::is_same<T, y>()
#define CASE2(x, y1, y2) \
    case x:              \
        return std::is_same<T, y1>() || std::is_same<T, y2>()
        switch (precisionInfo.value) {
            CASE(FP32, float);
            CASE2(FP16, int16_t, uint16_t);
            CASE(I16, int16_t);
            CASE(I32, int32_t);
            CASE(I64, int64_t);
            CASE(U16, uint16_t);
            CASE(U8, uint8_t);
            CASE(I8, int8_t);
            CASE(BOOL, uint8_t);
            CASE2(Q78, int16_t, uint16_t);
            CASE2(BIN, int8_t, uint8_t);
       default:
           return areSameStrings(name(), typeName == nullptr ? typeid(T).name() : typeName);
#undef CASE
#undef CASE2
            }
    } catch (...) {
        return false;
    }
}
template Precision Precision::fromType<int>(const char* typeName);
template Precision Precision::fromType<signed char>(const char* typeName);
template Precision Precision::fromType<short>(const char* typeName);
#if defined(ENABLE_GNA)
template Precision Precision::fromType<_compound_bias_t>(const char* typeName);
#endif // ENABLE_GNA
template bool Precision::hasStorageType<float>(const char* typeName) const;
template bool Precision::hasStorageType<int>(const char* typeName) const;
template bool Precision::hasStorageType<unsigned char>(const char* typeName) const;
template bool Precision::hasStorageType<short>(const char* typeName) const;
template bool Precision::hasStorageType<unsigned short>(const char* typeName) const;
template bool Precision::hasStorageType<signed char>(const char* typeName) const;
template bool Precision::hasStorageType<long>(const char* typeName) const;
template bool Precision::hasStorageType<unsigned long>(const char* typeName) const;
template bool Precision::hasStorageType<unsigned int>(const char* typeName) const;
template bool Precision::hasStorageType<double>(const char* typeName) const;
#if defined(ENABLE_GNA)
template bool Precision::hasStorageType<_compound_bias_t>(const char* typeName) const;
#endif // ENABLE_GNA

#endif // __clang__
