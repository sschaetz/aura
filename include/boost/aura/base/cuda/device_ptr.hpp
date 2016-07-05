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

template <typename T>
struct device_ptr_base_type
{
        CUdeviceptr device_buffer;

        /// Emulate memory_ = 0; behaviour of other base types.
        device_ptr_base_type& operator=(int a)
        {
                if (a == 0)
                {
                        device_buffer = nullptr;
                }
                return *this;
        }

        /// Access host ptr.
        T* get_host_ptr() { return nullptr; }
        const T* get_host_ptr() const { return nullptr; }

        /// Comparison operators
        bool operator==(const device_ptr_base_type<T>& other) const
        {
                return device_buffer == other.device_buffer;
        }

        bool operator!=(const device_ptr_base_type<T>& other) const
        {
                return !(*this == other);
        }

        /// Indicate if memory hold by pointer is shared with host or not.
        const bool is_shared_memory() const { return false; }
};



/// Specialize base_device_ptr for specific backend.
template <typename T>
using device_ptr =
        boost::aura::detail::base_device_ptr<T, device_ptr_base_type<T>>;


/// Allocate device memory.
template <typename T>
device_ptr<T> device_malloc(std::size_t size, device& d,
        memory_access_tag tag = memory_access_tag::rw)
{
        d.activate();
        typename device_ptr<T>::base_type m;
        AURA_CUDA_SAFE_CALL(cuMemAlloc(&m.device_buffer, size * sizeof(T)));
        d.deactivate();
        return device_ptr<T>(m, d, tag);
}

/// Free device memory.
template <typename T>
void device_free(device_ptr<T>& ptr)
{
        ptr.get_device().activate();
        AURA_CUDA_SAFE_CALL(cuMemFree(ptr.get_base_ptr().device_buffer));
        ptr.get_device().deactivate();
        ptr.reset();
}

} // cuda
} // base_detail
} // aura
} // boost
