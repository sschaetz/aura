#pragma once

#include <boost/aura/base/metal/device_ptr.hpp>
#include <boost/aura/base/metal/feed.hpp>

#include <iterator>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace metal
{

namespace detail
{

template <typename T>
T* unwrap(device_ptr<T>& ptr)
{
        return ptr.get_base_ptr().host_ptr.get() + ptr.get_offset();
}

template <typename T>
const T* unwrap(const device_ptr<T>& ptr)
{
        return const_cast<T*>(ptr.get_base_ptr().host_ptr.get()) +
                ptr.get_offset();
}

} // detail

/// Copy host memory to device.
template <typename InputIt, typename T>
void copy(InputIt first, InputIt last, device_ptr<T>& dst_first, feed& f)
{
        wait_for(f);
        std::copy(first, last, detail::unwrap(dst_first));
}


/// Copy device memory to host.
template <typename T, typename OutputIt>
void copy(const device_ptr<T>& first, const device_ptr<T>& last,
        OutputIt dst_first, feed& f)
{
        wait_for(f);
        std::copy(detail::unwrap(first), detail::unwrap(last), dst_first);
}

/// Copy device to device memory.
template <typename T>
void copy(const device_ptr<T>& first, const device_ptr<T>& last,
        device_ptr<T>& dst_first, feed& f)
{
        wait_for(f);
        std::copy(detail::unwrap(first), detail::unwrap(last),
                detail::unwrap(dst_first));
}

} // metal
} // base_detail
} // aura
} // boost
