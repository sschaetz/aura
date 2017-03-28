#pragma once

#include <boost/core/ignore_unused.hpp>

#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>

namespace boost
{
namespace aura
{

/// General allocator for device memory.
template <class T>
struct device_allocator
{
        using value_type = T;
        using pointer = device_ptr<T>;
        using const_pointer = const device_ptr<T>;

        /// Construct allocator.
        device_allocator(device& d)
                : device_(d)
        {}

        /// Copy construct allocator.
        template <class U>
        device_allocator(const device_allocator<U>& other)
                : device_(other.device_)
        {}

        /// Allocate memory.
        pointer allocate(std::size_t n)
        {
                return device_malloc<T>(n, device_);
        }

        /// Deallocate memory.
        void deallocate(pointer& p, std::size_t n)
        {
                boost::ignore_unused(n);
                device_free(p);
        }

private:
        /// Device we allocate memory from.
        device& device_;
};

} // namespace aura
} // namespace boost
