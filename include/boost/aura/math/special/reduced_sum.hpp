#ifndef AURA_MATH_SPECIAL_REDUCED_SUM_HPP
#define AURA_MATH_SPECIAL_REDUCED_SUM_HPP

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

inline std::tuple<const char*,const char*> get_reduced_sum_kernel(cfloat, cfloat)
{
	return std::make_tuple("reduced_sum_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void reduced_sum_cfloat(AURA_GLOBAL cfloat* src,
			AURA_GLOBAL cfloat* dst,
			unsigned long N,
			unsigned long n)
	{
		int dimensions = N/n; 
		unsigned int i = get_mesh_id();

        if (i < n)
            dst[i] = make_cfloat(0.,0.);

        for (int j = 0; j < dimensions; j++) {
			if (i < n) {
				dst[i] += src[i+j*n];
			}
                }
	}
		
		)aura_kernel");	
}

} // namespace detail


template <typename DeviceRangeType1, typename DeviceRangeType2>
void reduced_sum(const DeviceRangeType1& input_range, 
		DeviceRangeType2& output_range, feed& f)
{
      
        // and vectors life on the same device
	assert(aura::traits::get_device(input_range) == 
			aura::traits::get_device(output_range));
	// deactivate these asserts by defining NDEBUG	

	auto kernel_data = detail::get_reduced_sum_kernel(
			aura::traits::get_value_type(input_range),
			aura::traits::get_value_type(output_range));

	backend::kernel k = aura::traits::get_device(output_range).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS, true);

	invoke(k, aura::traits::bounds(input_range), 
			args(aura::traits::data(input_range),
				aura::traits::data(output_range),
				aura::traits::size(input_range), 
				aura::traits::size(output_range)), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_SPECIAL_REDUCED_SUM_HPP

