#ifndef AURA_COPY_HPP
#define AURA_COPY_HPP

#include <boost/aura/backend.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/device_range.hpp>

namespace boost
{
namespace aura 
{

template <typename T>
class device_array;

template <typename T>
class device_range;

/// copy to device array
template <typename T>
void copy(const T* src, device_array<T>& dst, backend::feed& f) 
{
	backend::copy<T>(dst.begin(), src, dst.size(), f);
}

/// copy from device array
template <typename T>
void copy(const device_array<T>& src, T* dst, feed& f) 
{
	backend::copy<T>(dst, src.begin(), src.size(), f);
}

/// copy from std::vector to device array
template <typename T>
void copy(const std::vector<T>& src, device_array<T>& dst, feed& f)
{
	backend::copy<T>(dst.begin(), &src[0], dst.size(), f);
}

/// copy from device array to std::vector
template <typename T>
void copy(const device_array<T>& src, std::vector<T>& dst, feed& f)
{
	backend::copy<T>(&dst[0], src.begin(), src.size(), f);
}

/// copy to device array
template <typename T>
void copy(const T* src, device_range<T>& dst, backend::feed& f) 
{
	backend::copy<T>(dst.begin(), src, dst.size(), f);
}

/// copy from device array
template <typename T>
void copy(const device_range<T>& src, T* dst, feed& f) 
{
	backend::copy<T>(dst, src.begin(), src.size(), f);
}

/// copy from std::vector to device array
template <typename T>
void copy(const std::vector<T>& src, device_range<T>& dst, feed& f)
{
	backend::copy<T>(dst.begin(), &src[0], dst.size(), f);
}

/// copy from device array to std::vector
template <typename T>
void copy(const device_range<T>& src, std::vector<T>& dst, feed& f)
{
	backend::copy<T>(&dst[0], src.begin(), src.size(), f);
}

/// copy from device array to device array
template <typename T>
void copy(const device_array<T>& src, device_array<T>& dst, feed& f)
{
	backend::copy<T>(dst.begin(), src.begin(), src.size(), f);
}




/// copy from and to device ranges
template <typename T>
void copy(const device_array<T>& src, device_range<T>& dst, feed& f)
{
    backend::copy<T>(dst.begin(), src.begin(), src.size(), f);
}
template <typename T>
void copy(const device_range<T>& src, device_array<T>& dst, feed& f)
{
    backend::copy<T>(dst.begin(), src.begin(), src.size(), f);
}
template <typename T>
void copy(const device_range<T>& src, device_range<T>& dst, feed& f)
{
    backend::copy<T>(dst.begin(), src.begin(), src.size(), f);
}

} // namespace aura
} // boost

#endif // AURA_COPY_HPP

