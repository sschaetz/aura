#ifndef AURA_MATH_NORM2_HPP
#define AURA_MATH_NORM2_HPP

#include <tuple>

#include <boost/aura/meta/traits.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/math/complex.hpp>

// TODO: solve ambiguity of type (type changes by user) through wrapper function
// just like conj()

namespace boost
{
namespace aura
{
namespace math
{

namespace detail
{

inline std::tuple<const char*,const char*> norm2_kernel_name(
		device_ptr<float> src_ptr, 
		device_ptr<float> dst_ptr)
{
	return std::make_tuple("norm2_float",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void norm2_float(AURA_GLOBAL float* src_ptr, 
			AURA_GLOBAL float* dst_ptr)
	{
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> norm2_kernel_name(
		device_ptr<cfloat> src_ptr, 
		device_ptr<cfloat> dst_ptr)
{
	return std::make_tuple("norm2_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void norm2_float(AURA_GLOBAL float* src_ptr, 
			AURA_GLOBAL float* dst_ptr)
	{
	}
		
		)aura_kernel");	
}

}

template <typename DeviceRangeType>
void norm2(DeviceRangeType& input_range, 
		DeviceRangeType& output_range, feed& f)
{
	auto kernel_data = detail::norm2_kernel_name(
			aura::traits::begin(input_range),
			aura::traits::begin(output_range));
	backend::kernel k = aura::traits::get_device(input_range).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS);
	invoke(k, aura::traits::size(input_range), 
			args(aura::traits::begin(input_range), 
				aura::traits::begin(output_range)), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_NORM2_HPP

