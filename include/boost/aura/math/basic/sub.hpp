#ifndef AURA_MATH_BASIC_SUB_HPP
#define AURA_MATH_BASIC_SUB_HPP

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

inline std::tuple<const char*,const char*> get_sub_kernel(float, float, float)
{
	return std::make_tuple("sub_float",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void sub_float(AURA_GLOBAL float* src1,
			AURA_GLOBAL float* src2,
			AURA_GLOBAL float* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = src1[i] - src2[i];
		}		
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_sub_kernel(cfloat, cfloat, cfloat)
{
	return std::make_tuple("sub_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void sub_cfloat(AURA_GLOBAL cfloat* src1,
			AURA_GLOBAL cfloat* src2,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = csubf(src1[i], src2[i]);
		}		
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_sub_kernel(float, cfloat, cfloat)
{
	return std::make_tuple("sub_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void sub_cfloat(AURA_GLOBAL float* src1,
			AURA_GLOBAL cfloat* src2,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = make_cfloat(src1[i]-src2[i].x, -src2[i].y);
			//dst[i] = caddf(src1[i], src2[i]);
		}		
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_sub_kernel(cfloat, float, cfloat)
{
	return std::make_tuple("sub_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void sub_cfloat(AURA_GLOBAL cfloat* src1,
			AURA_GLOBAL float* src2,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = make_cfloat(src1[i].x-src2[i], src1[i].y);
		}		
	}
		
		)aura_kernel");	
}


} // namespace detail

template <typename DeviceRangeType1, 
	 typename DeviceRangeType2, 
	typename DeviceRangeType3>
void sub(const DeviceRangeType1& input_range1,
		const DeviceRangeType2& input_range2,
		DeviceRangeType3& output_range, feed& f)
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

	auto kernel_data = detail::get_sub_kernel(
			aura::traits::get_value_type(input_range1),
			aura::traits::get_value_type(input_range2),
			aura::traits::get_value_type(output_range));

	backend::kernel k = aura::traits::get_device(output_range).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS, true);

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

#endif // AURA_MATH_BASI_SUB_HPP

