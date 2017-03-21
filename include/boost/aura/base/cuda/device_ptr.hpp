#pragma once

#include <boost/aura/base/base_device_ptr.hpp>
#include <boost/aura/base/cuda/device.hpp>
#include <boost/aura/base/cuda/feed.hpp>
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

        void reset()
        {}

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
        bool is_shared_memory() const { return false; }
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
        auto size_bytes = size * sizeof(T);
        AURA_CUDA_SAFE_CALL(cuMemAlloc(&m.device_buffer, size_bytes));
        d.deactivate();
        d.allocation_tracker.add(m.device_buffer, size_bytes);
        return device_ptr<T>(m, d, tag, d.is_shared_memory());
}

/// Free device memory.
template <typename T>
void device_free(device_ptr<T>& ptr)
{
        ptr.get_device().activate();
        auto buffer = ptr.get_base_ptr().device_buffer;
        ptr.get_device().allocation_tracker.remove(buffer);
        AURA_CUDA_SAFE_CALL(cuMemFree(buffer));
        ptr.get_device().deactivate();
        ptr.reset();
}

/// Set device memory (bytes).
template <typename T>
void device_memset(device_ptr<T>& ptr, char value, std::size_t num, feed& f)
{
        ptr.get_device().activate();
        AURA_CUDA_SAFE_CALL(
                        cuMemsetD8(
                                ptr.get_base_ptr().device_buffer,
                                value,
                                num
                        )
                );
        wait_for(f);
        ptr.get_device().deactivate();
}

} // cuda
} // base_detail
} // aura
} // boost
