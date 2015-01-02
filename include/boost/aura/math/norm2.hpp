#ifndef AURA_MATH_NORM2_HPP
#define AURA_MATH_NORM2_HPP

#include <boost/aura/meta/traits.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/math/complex.hpp>

namespace boost
{
namespace aura
{
namespace math
{

namespace detail
{

inline const char* norm2_kernel_name(device_ptr<float> src_ptr, 
		device_ptr<float> dst_ptr)
{
	return "norm2_float";	
}

inline const char* norm2_kernel_name(device_ptr<cfloat> src_ptr, 
		device_ptr<cfloat> dst_ptr)
{
	return "norm2_cfloat";	
}

}

template <typename DeviceRangeType>
void norm2(DeviceRangeType& input_range, 
		DeviceRangeType& output_range, feed& f)
{
	auto kernel_name = detail::norm2_kernel_name(
			aura::traits::begin(input_range),
			aura::traits::begin(output_range));
	backend::kernel k = src_ptr.get_device().load_from_file(
			"norm2_kernels.cc", kernel_name,
			AURA_BACKEND_COMPILE_FLAGS);
	invoke(k, size, args(aura::traits::begin(input_range), 
				aura::traits::begin(output_range)), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_NORM2_HPP

