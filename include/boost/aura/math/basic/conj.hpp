#ifndef AURA_MATH_BASIC_CONJ_HPP
#define AURA_MATH_BASIC_CONJ_HPP

#include <tuple>
#include <cassert>

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

inline std::tuple<const char*,const char*> get_conj_kernel(cfloat, cfloat)
{
	return std::make_tuple("conj_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void conj_cfloat(AURA_GLOBAL cfloat* src,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
                       dst[i] = conjf(src[i]); 
		}		
	}
		
		)aura_kernel");	
}

} // namespace detail


template <typename DeviceRangeType1, typename DeviceRangeType2>
void conj(const DeviceRangeType1& input_range, 
		DeviceRangeType2& output_range, feed& f)
{
	// asserts to make sure vectors have same size
	assert(aura::traits::size(input_range) ==
			aura::traits::size(output_range));
      
        // and vectors life on the same device
	assert(aura::traits::get_device(input_range) ==
			aura::traits::get_device(output_range));
	
	// deactivate these asserts by defining NDEBUG
	auto kernel_data = detail::get_conj_kernel(
			aura::traits::get_value_type(input_range),
			aura::traits::get_value_type(output_range));

	backend::kernel k = aura::traits::get_device(output_range).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS, true);

	invoke(k, aura::traits::bounds(input_range),
			args(aura::traits::begin_raw(input_range),
				aura::traits::begin_raw(output_range),
				aura::traits::size(input_range)), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_BASIC_CONJ_HPP

