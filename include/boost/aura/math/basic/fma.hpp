#ifndef AURA_MATH_BASIC_FMA_HPP
#define AURA_MATH_BASIC_FMA_HPP

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

inline std::tuple<const char*,const char*> get_fma_kernel(float, float, float)
{
	return std::make_tuple("fma_float",
			R"aura_kernel(

	#include <boost/aura/backend.hpp>

	AURA_KERNEL void fma_float(AURA_GLOBAL float* src1,
			AURA_GLOBAL float* src2,
			AURA_GLOBAL float* src3,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
                       src3[i] = fmaf(src1[i],src2[i],src3[i]);
		}
	}

		)aura_kernel");
}

inline std::tuple<const char*,const char*> get_fma_kernel(cfloat, cfloat, cfloat)
{
	return std::make_tuple("fma_cfloat",
			R"aura_kernel(

	#include <boost/aura/backend.hpp>

	AURA_KERNEL void fma_cfloat(AURA_GLOBAL cfloat* src1,
			AURA_GLOBAL cfloat* src2,
			AURA_GLOBAL cfloat* src3,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
                       src3[i] = caddf(cmulf(src1[i],src2[i]),src3[i]);
		}
	}

		)aura_kernel");
}

} // namespace detail


template <typename DeviceRangeType>
void fma(const DeviceRangeType& input_range1,
		const DeviceRangeType& input_range2,
		DeviceRangeType& input_output_range, feed& f)
{
	// asserts to make sure vectors have same size
	assert(aura::traits::size(input_range1) == 
			aura::traits::size(input_range2));
	assert(aura::traits::size(input_range2) ==
			aura::traits::size(input_output_range));
        // and vectors life on the same device
	assert(aura::traits::get_device(input_range1) == 
			aura::traits::get_device(input_range2));
	assert(aura::traits::get_device(input_range2) ==
			aura::traits::get_device(input_output_range));
	// deactivate these asserts by defining NDEBUG

	auto kernel_data = detail::get_fma_kernel(
			aura::traits::get_value_type(input_range1),
			aura::traits::get_value_type(input_range2),
			aura::traits::get_value_type(input_output_range));

	backend::kernel k = aura::traits::get_device(input_output_range).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS, true);

	invoke(k, aura::traits::bounds(input_range2), 
			args(aura::traits::data(input_range1), 
				aura::traits::data(input_range2),
				aura::traits::data(input_output_range),
				aura::traits::size(input_range2)), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_BASIC_FMA_HPP

