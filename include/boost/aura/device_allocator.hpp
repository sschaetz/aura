#pragma once

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

        /// Construct empty allocator.
        device_allocator()
        {}

        /// Construct allocator.
        device_allocator(device& d)
                : device_(&d)
        {}

        /// Copy construct allocator.
        template <class U>
        device_allocator(const device_allocator<U>& other)
                : device_(other.device_)
        {}

        /// Move construct allocator.
        template <class U>
        device_allocator(device_allocator<U>&& other)
                : device_(other.device_)
        {
                other.device_ = nullptr;
        }

        /// Allocate memory.
        pointer allocate(std::size_t n)
        {
                assert(device_);
                return device_malloc<T>(n, *device_);
        }

        /// Deallocate memory.
        void deallocate(pointer& p, std::size_t)
        {
                assert(device_);
                device_free(p);
        }

private:
        /// Device we allocate memory from.
        device* device_;

        /// Flag that indicates if allocator is initialized or not.
        bool initialized_ { false };
};

} // namespace aura
} // namespace boost
