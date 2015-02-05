#ifndef AURA_META_TRAITS_HPP
#define AURA_META_TRAITS_HPP

#include <boost/aura/device_array.hpp>

namespace boost
{
namespace aura
{
namespace traits
{

template <typename T>
device_ptr<T> begin(device_array<T>& da)
{
	return da.begin();
}

template <typename T>
T* begin_raw(device_array<T>& da)
{
	return (T*)da.begin().get();
}

template <typename T>
const T* begin_raw(const device_array<T>& da)
{
	return (T*)da.begin().get();
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

} // namespace traits
} // namespace aura
} // namespace boost

#endif // AURA_META_TRAITS_HPP

