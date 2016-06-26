#pragma once

#include <boost/aura/base/base_device_ptr.hpp>
#include <boost/aura/base/cuda/device.hpp>
#include <boost/aura/memory_tag.hpp>

#include <cuda.h>

#include <cstddef>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace cuda
{

/// Specialize base_device_ptr for specific backend.
template <typename T>
using device_ptr = boost::aura::detail::base_device_ptr<T, CUdeviceptr>;

/// equal to operator (reverse order)
template <typename T>
bool operator==(std::nullptr_t, const device_ptr<T> &ptr)
{
        return (ptr == nullptr);
}

/// not equal to operator (reverse order)
template <typename T>
bool operator!=(std::nullptr_t, const device_ptr<T> &ptr)
{
        return (ptr != nullptr);
}


/// Allocate device memory.
template <typename T>
device_ptr<T> device_malloc(std::size_t size, device &d,
        memory_access_tag tag = memory_access_tag::rw)
{
        d.activate();
        typename device_ptr<T>::base_type m;
        AURA_CUDA_SAFE_CALL(cuMemAlloc(&m, size * sizeof(T)));
        d.deactivate();
        return device_ptr<T>(m, d, tag);
}

/// Free device memory.
template <typename T> void device_free(device_ptr<T> &ptr)
{
        ptr.get_device().activate();
        AURA_CUDA_SAFE_CALL(cuMemFree(ptr.get_base_ptr()));
        ptr.get_device().deactivate();
        ptr.reset();
}

} // cuda
} // base_detail
} // aura
} // boost
