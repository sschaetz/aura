#ifndef AURA_MATH_ADD_HPP
#define AURA_MATH_ADD_HPP

#include <tuple>
#include <cassert>

#include <boost/aura/meta/traits.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/math/complex.hpp>
#include <boost/aura/math/add_kernel.hpp>

namespace boost
{
namespace aura
{
namespace math
{


template <typename DeviceRangeType>
void add(DeviceRangeType& input_range1, 
		DeviceRangeType& input_range2,
		DeviceRangeType& output_range, feed& f)
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

	auto kernel_data = detail::get_add_kernel(
			aura::traits::get_value_type(input_range1),
			aura::traits::get_value_type(input_range2),
			aura::traits::get_value_type(output_range));

	backend::kernel k = aura::traits::get_device(input_range1).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS);
	invoke(k, aura::traits::bounds(input_range1), 
			args(aura::traits::begin_raw(input_range1), 
				aura::traits::begin_raw(input_range2),
				aura::traits::begin_raw(output_range),
				aura::traits::size(input_range1)), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_ADD_HPP

