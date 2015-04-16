#ifndef AURA_META_TRAITS_HPP
#define AURA_META_TRAITS_HPP

#include <boost/aura/device_array.hpp>
#include <boost/aura/device_range.hpp>

namespace boost
{
namespace aura
{
namespace traits
{

// device array traits
template <typename T>
device_ptr<T> begin(device_array<T>& da)
{
	return da.begin();
}

template <typename T>
T* data(device_array<T>& da)
{
	return (T*)da.data();
}

template <typename T>
const T* data(const device_array<T>& da)
{
	return (T*)da.data();
}

template <typename T>
std::size_t size(const device_array<T>& da)
{
	return da.size();
}

template <typename T>
bounds bounds(const device_array<T>& da)
{
	return da.get_bounds();
}

template <typename T>
device& get_device(device_array<T>& da)
{
	return da.get_device();
}

template <typename T>
const device& get_device(const device_array<T>& da)
{
	return da.get_device();
}

template <typename T>
T get_value_type(device_array<T>& da)
{
	return T();
}

template <typename T>
T get_value_type(const device_array<T>& da)
{
	return T();
}

// device range traits
template <typename T>
device_ptr<T> begin(device_range<T>& dr)
{
	return dr.begin();
}

template <typename T>
T* data(device_range<T>& dr)
{
	return (T*)dr.data();
}

template <typename T>
const T* data(const device_range<T>& dr)
{
	return (T*)dr.data();
}

template <typename T>
std::size_t size(const device_range<T>& dr)
{
	return dr.size();
}

template <typename T>
boost::aura::bounds bounds(const device_range<T>& dr)
{
    return dr.get_bounds();
}

template <typename T>
device& get_device(device_range<T>& dr)
{
	return dr.get_device();
}

template <typename T>
const device& get_device(const device_range<T>& dr)
{
	return dr.get_device();
}

template <typename T>
T get_value_type(device_range<T>& dr)
{
	return T();
}

template <typename T>
T get_value_type(const device_range<T>& dr)
{
	return T();
}


} // namespace traits
} // namespace aura
} // namespace boost

#endif // AURA_META_TRAITS_HPP

