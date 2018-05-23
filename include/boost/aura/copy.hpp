#pragma once

#include <boost/aura/device_array.hpp>
#include <boost/aura/feed.hpp>

#if defined AURA_BASE_CUDA
#include <boost/aura/base/cuda/copy.hpp>
#elif defined AURA_BASE_OPENCL
#include <boost/aura/base/opencl/copy.hpp>
#elif defined AURA_BASE_METAL
#include <boost/aura/base/metal/copy.hpp>
#endif

namespace boost
{
namespace aura
{

#if defined AURA_BASE_CUDA
namespace base = base_detail::cuda;
#elif defined AURA_BASE_OPENCL
namespace base = base_detail::opencl;
#elif defined AURA_BASE_METAL
namespace base = base_detail::metal;
#endif

/// copy to device array from an iterator
template <typename Iterator,
        typename T,
        typename Allocator,
        typename BoundsType
>
void copy(Iterator src,
        device_array<T, Allocator, BoundsType>& dst,
        base::feed& f
)
{
        typedef typename std::iterator_traits<Iterator>::value_type T2;
        static_assert(std::is_same<T, T2>::value,
                "iterator value type and device_array type must match");
        base::copy(&(*src), &(*src) + dst.size(), dst.begin(), f);
}

/// copy to device array
template <typename T,
        typename Allocator,
        typename BoundsType
>
void copy(const T* src,
        device_array<T, Allocator, BoundsType>& dst,
        base::feed& f
)
{
        base::copy(src, src + dst.size(), dst.begin(), f);
}

/// copy from device array
template <typename T,
        typename Allocator,
        typename BoundsType
>
void copy(const device_array<T, Allocator, BoundsType>& src,
        T* dst,
        feed& f
)
{
        base::copy(src.begin(), src.end(), dst, f);
}

/// copy from std::vector to device array
template <typename T,
        typename Allocator,
        typename BoundsType
>
void copy(const std::vector<T>& src,
        device_array<T, Allocator, BoundsType>& dst,
        feed& f
)
{
        base::copy(src.begin(), src.end(), dst.begin(), f);
}

/// copy from std::vector to device ptr
template <typename T>
void copy(const std::vector<T>& src, const device_ptr<T>& dst, feed& f)
{
        base::copy(src.begin(), src.end(), dst, f);
}

/// copy from device array to std::vector
template <typename T,
        typename Allocator,
        typename BoundsType
>
void copy(const device_array<T, Allocator, BoundsType>& src,
        std::vector<T>& dst,
        feed& f
)
{
        base::copy(src.begin(), src.end(), &dst[0], f);
}

/// copy from device ptr to std::vector
template <typename T>
void copy(const device_ptr<T>& src, std::vector<T>& dst, feed& f)
{
        base::copy(src, src + dst.size(), &dst[0], f);
}

/// copy from device array to device array
template <typename T,
        typename Allocator,
        typename BoundsType
>
void copy(const device_array<T, Allocator, BoundsType>& src,
        device_array<T, Allocator, BoundsType>& dst,
        feed& f
)
{
        base::copy(src.begin(), src.end(), dst.begin(), f);
}

using base::copy;

} // namespace aura
} // namespace boost
