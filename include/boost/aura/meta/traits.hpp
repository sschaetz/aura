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
using value_type = typename aura::device_array<T>::value_type;

template <typename T>
device_ptr<T> begin(device_array<T>& da)
{
	return da.begin();
}

template <typename T>
std::size_t size(device_array<T>& da)
{
	return da.size();
}

template <typename T>
device& get_device(device_array<T>& da)
{
	return da.get_device();
}

} // namespace traits
} // namespace aura
} // namespace boost

#endif // AURA_META_TRAITS_HPP

