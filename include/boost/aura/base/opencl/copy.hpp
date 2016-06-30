#pragma once

#include <boost/aura/base/opencl/device_ptr.hpp>
#include <boost/aura/base/opencl/feed.hpp>

#include <iterator>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace opencl
{

/// Copy host memory to device.
template <typename InputIt, typename T>
void copy(InputIt first, InputIt last, device_ptr<T> dst_first, feed& f)
{
        AURA_OPENCL_SAFE_CALL(clEnqueueWriteBuffer(f.get_base_feed(),
                dst_first.get_base_ptr(), CL_FALSE,
                dst_first.get_offset() * sizeof(T),
                std::distance(first, last) * sizeof(T), &(*first), 0, NULL,
                NULL));
}


/// Copy device memory to host.
template <typename T, typename OutputIt>
void copy(const device_ptr<T> first, const device_ptr<T> last,
        OutputIt dst_first, feed& f)
{
        AURA_OPENCL_SAFE_CALL(clEnqueueReadBuffer(f.get_base_feed(),
                first.get_base_ptr(), CL_FALSE, first.get_offset() * sizeof(T),
                std::distance(first, last) * sizeof(T), &(*dst_first), 0, NULL,
                NULL));
}

/// Copy device to device memory.
template <typename T>
void copy(const device_ptr<T> first, const device_ptr<T> last,
        device_ptr<T> dst_first, feed& f)
{
        AURA_OPENCL_SAFE_CALL(clEnqueueCopyBuffer(f.get_base_feed(),
                first.get_base_ptr(), dst_first.get_base_ptr(),
                first.get_offset() * sizeof(T),
                dst_first.get_offset() * sizeof(T),
                std::distance(first, last) * sizeof(T), 0, 0, 0));
}

} // opencl
} // base_detail
} // aura
} // boost
