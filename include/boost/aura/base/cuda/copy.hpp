#pragma once

#include <boost/aura/base/cuda/device_ptr.hpp>
#include <boost/aura/base/cuda/feed.hpp>

#include <cuda.h>

#include <iterator>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace cuda
{

/// Copy host memory to device.
template <typename InputIt, typename T>
void copy(InputIt first, InputIt last, device_ptr<T> dst_first, feed& f)
{
        f.get_device().activate();
        AURA_CUDA_SAFE_CALL(cuMemcpyHtoDAsync(
                dst_first.get_base_ptr() + dst_first.get_offset() * sizeof(T),
                &(*first), std::distance(first, last) * sizeof(T),
                f.get_base_feed()));
        f.get_device().deactivate();
}


/// Copy device memory to host.
template <typename T, typename OutputIt>
void copy(const device_ptr<T> first, const device_ptr<T> last,
        OutputIt dst_first, feed& f)
{
        f.get_device().activate();
        AURA_CUDA_SAFE_CALL(cuMemcpyDtoHAsync(&(*dst_first),
                first.get_base_ptr() + first.get_offset(),
                std::distance(first, last) * sizeof(T), f.get_base_feed()));
        f.get_device().deactivate();
}

/// Copy device to device memory.
template <typename T>
void copy(const device_ptr<T> first, const device_ptr<T> last,
        device_ptr<T> dst_first, feed& f)
{
        f.get_device().activate();
        AURA_CUDA_SAFE_CALL(cuMemcpyDtoDAsync(
                dst_first.get_base_ptr() + dst_first.get_offset() * sizeof(T),
                first.get_base_ptr() + first.get_offset() * sizeof(T),
                std::distance(first, last) * sizeof(T), f.get_base_feed()));
        f.get_device().deactivate();
}

} // cuda
} // base_detail
} // aura
} // boost
