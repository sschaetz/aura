#pragma once

#include <boost/aura/base/base_device_ptr.hpp>
#include <boost/aura/base/metal/device.hpp>
#include <boost/aura/memory_tag.hpp>

#include <cstddef>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace metal
{

template <typename T>
struct device_ptr_base_type
{
        id<MTLBuffer> device_buffer;
        std::shared_ptr<T> host_ptr;

        // Emulate memory_ = 0; behaviour of other base types.
        device_ptr_base_type &operator=(int a)
        {
                if (a == 0)
                {
                        device_buffer = nil;
                        host_ptr.reset();
                }
                return *this;
        }
};


template <typename T>
using device_ptr =
        boost::aura::detail::base_device_ptr<T, device_ptr_base_type<T>>;

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

namespace detail
{

void free_posix_memalign(void *ptr, NSUInteger)
{
        free(ptr);
}

} // detail

int counter = 0;

/// Allocate device memory.
template <typename T>
device_ptr<T> device_malloc(std::size_t size, device &d,
        memory_access_tag tag = memory_access_tag::rw)
{
        constexpr std::size_t metal_memory_alignment = 16384;
        std::size_t num_bytes = size * sizeof(T);
        // Compute aligned array size.
        size_t aligned_size = num_bytes +
                (metal_memory_alignment - (num_bytes % metal_memory_alignment));

        void *host_ptr;

        // Allocate array.
        int err = 0;
        err = posix_memalign(&host_ptr, metal_memory_alignment, aligned_size);
        AURA_METAL_CHECK_ERROR((err == KERN_SUCCESS));

        // Create buffer.
        typename device_ptr<T>::base_type m;
        m.device_buffer =
                [d.get_base_device() newBufferWithBytesNoCopy:host_ptr
                                                       length:aligned_size
                                                      options:0
                                                  deallocator:nil];
        m.host_ptr =
                std::shared_ptr<T>(reinterpret_cast<T *>(host_ptr), [](T *ptr)
                        {
                                free(ptr);
                        });

        AURA_METAL_CHECK_ERROR(m.device_buffer);
        return device_ptr<T>(m, d, tag);
}

/// Free device memory.
template <typename T>
void device_free(device_ptr<T> &ptr)
{
        ptr.reset();
}


} // metal
} // base_detail
} // aura
} // boost
