#ifndef AURA_MATH_COMPLEX_INTERLEAVED_HPP
#define AURA_MATH_COMPLEX_INTERLEAVED_HPP

#include <tuple>
#include <cassert>

#include <boost/aura/meta/traits.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/math/complex.hpp>

// #include "split_interleaved_s2i_kernels.hpp"
// #include "split_interleaved_i2s_kernels.hpp"

namespace boost
{
namespace aura
{
namespace math
{

namespace detail
{
  inline std::tuple<const char*,const char*> get_s2i_kernel(float, float, cfloat)
  {
    return std::make_tuple("s2i_float",split_interleaved_s2i_kernels);
  }


  inline std::tuple<const char*,const char*> get_i2s_kernel(cfloat, float, float)
  {
    return std::make_tuple("i2s_float",split_interleaved_i2s_kernels);
  }

} // namespace detail


template <typename DeviceRangeType, typename DeviceRangeTypeComplex>
void s2i(const DeviceRangeType& input_range1,
		const DeviceRangeType& input_range2,
		DeviceRangeTypeComplex& output_range, feed& f)
{
	// asserts to make sure vectors have same size
	assert(aura::traits::size(input_range1) == 
			aura::traits::size(input_range2));
	assert(aura::traits::size(input_range1) == 
			aura::traits::size(output_range));
	// and vectors life on the same device
	assert(aura::traits::get_device(input_range1) == 
			aura::traits::get_device(input_range2));
	assert(aura::traits::get_device(input_range1) == 
			aura::traits::get_device(output_range));
	// deactivate these asserts by defining NDEBUG	

	auto kernel_data = detail::get_s2i_kernel(
			aura::traits::get_value_type(input_range1),
			aura::traits::get_value_type(input_range2),
			aura::traits::get_value_type(output_range));

	backend::kernel k = aura::traits::get_device(output_range).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS);

	invoke(k, aura::traits::bounds(input_range1), 
			args(aura::traits::data(input_range1), 
				aura::traits::data(input_range2),
				aura::traits::data(output_range),
				aura::traits::size(input_range1)), f);
	return;
}

template <typename DeviceRangeType, typename DeviceRangeTypeComplex>
void i2s(const DeviceRangeTypeComplex& input_range,
		DeviceRangeType& output_range1,
		DeviceRangeType& output_range2, feed& f)
{
	// asserts to make sure vectors have same size
	assert(aura::traits::size(input_range) ==
			aura::traits::size(output_range1));
	assert(aura::traits::size(output_range1) ==
			aura::traits::size(output_range2));
	// and vectors life on the same device
	assert(aura::traits::get_device(input_range) ==
			aura::traits::get_device(output_range1));
	assert(aura::traits::get_device(output_range1) ==
			aura::traits::get_device(output_range2));
	// deactivate these asserts by defining NDEBUG

	auto kernel_data = detail::get_i2s_kernel(
			aura::traits::get_value_type(input_range),
			aura::traits::get_value_type(output_range1),
			aura::traits::get_value_type(output_range2));

	backend::kernel k = aura::traits::get_device(output_range1).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS);

	invoke(k, aura::traits::bounds(input_range),
			args(aura::traits::data(input_range),
				aura::traits::data(output_range1),
				aura::traits::data(output_range2),
				aura::traits::size(input_range)), f);
	return;
}


} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_COMPLEX_INTERLEAVED_HPP

