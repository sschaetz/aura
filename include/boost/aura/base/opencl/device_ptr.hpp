#pragma once

#include <boost/aura/base/base_device_ptr.hpp>
#include <boost/aura/base/opencl/device.hpp>
#include <boost/aura/base/opencl/feed.hpp>
#include <boost/aura/memory_tag.hpp>


#include <cstddef>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace opencl
{

template <typename T>
struct device_ptr_base_type
{
        cl_mem device_buffer;

        // Emulate memory_ = 0; behaviour of other base types.
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

        bool operator==(const device_ptr_base_type<T>& other) const
        {
                return device_buffer == other.device_buffer;
        }

        bool operator!=(const device_ptr_base_type<T>& other) const
        {
                return !(*this == other);
        }

        /// @copydoc
        /// boost::aura::base::cuda::device_base_ptr::is_shared_memory()
        bool is_shared_memory() const { return false; }
};

/// Specialize base_device_ptr for specific backend.
template <typename T>
using device_ptr =
        boost::aura::detail::base_device_ptr<T, device_ptr_base_type<T>>;


/// equal to operator (reverse order)
template <typename T>
bool operator==(std::nullptr_t, const device_ptr<T>& ptr)
{
        return (ptr == nullptr);
}

/// not equal to operator (reverse order)
template <typename T>
bool operator!=(std::nullptr_t, const device_ptr<T>& ptr)
{
        return (ptr != nullptr);
}


/// Translates an Aura memory tag to an OpenCL memory tag.
inline cl_mem_flags translate_memory_access_tag(memory_access_tag tag)
{
        switch (tag)
        {
        case memory_access_tag::ro:
                return CL_MEM_READ_ONLY;
        case memory_access_tag::rw:
                return CL_MEM_READ_WRITE;
        case memory_access_tag::wo:
                return CL_MEM_WRITE_ONLY;
        default:
                return CL_MEM_READ_WRITE;
        }
}


/// Allocate device memory.
template <typename T>
device_ptr<T> device_malloc(std::size_t size, device& d,
        memory_access_tag tag = memory_access_tag::rw)
{
        int errorcode = 0;
        typename device_ptr<T>::base_type m;
        auto size_bytes = size * sizeof(T);
        m.device_buffer = clCreateBuffer(d.get_base_context(),
                translate_memory_access_tag(tag), size_bytes, 0,
                &errorcode);
        AURA_OPENCL_CHECK_ERROR(errorcode);
        d.allocation_tracker.add(m.device_buffer, size_bytes);
        return device_ptr<T>(m, d, tag, d.is_shared_memory());
}

/// Free device memory.
template <typename T>
void device_free(device_ptr<T>& ptr)
{
        auto buffer = ptr.get_base_ptr().device_buffer;
        ptr.get_device().allocation_tracker.remove(buffer);
        AURA_OPENCL_SAFE_CALL(clReleaseMemObject(buffer));
        ptr.reset();
}

/// Set device memory (bytes).
template <typename T>
void device_memset(device_ptr<T>& ptr, char value, std::size_t num, feed& f)
{
#ifdef CL_VERSION_1_2
        AURA_OPENCL_SAFE_CALL(
                clEnqueueFillBuffer(
                        f.get_base_feed(),
                        ptr.get_base_ptr().device_buffer,
                        reinterpret_cast<const void*>(&value),
                        1,
                        0,
                        num,
                        0,
                        NULL,
                        NULL
                )
        );
#else
        std::vector<char> tmp(num, value);
        AURA_OPENCL_SAFE_CALL(
                clEnqueueWriteBuffer(
                        f.get_base_feed(),
                        ptr.get_base_ptr().device_buffer,
                        CL_TRUE,
                        ptr.get_offset() * sizeof(T),
                        num,
                        &(tmp[0]),
                        0,
                        NULL,
                        NULL
                )
        );
#endif
}

} // opencl
} // base_detail
} // aura
} // boost
