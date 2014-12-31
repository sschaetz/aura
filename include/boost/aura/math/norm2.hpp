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

inline void norm2(device_ptr<float> src_ptr, const std::size_t size,
		device_ptr<float> dst_ptr, feed& f)
{
	const char* kernel_string = R"aura_kernel(
		#include <boost/aura/backend.hpp>
		AURA_KERNEL
		void norm2_float(AURA_GLOBAL float* src_ptr, 
				AURA_GLOBAL float* dst_ptr) {}
	)aura_kernel";	
	backend::kernel k = src_ptr.get_device().load_from_string(
			kernel_string, "norm2_float",
			AURA_BACKEND_COMPILE_FLAGS);
	invoke(k, size, args(src_ptr.get(), dst_ptr.get()), f);
}

inline void norm2(device_ptr<cfloat> src_ptr, const std::size_t size,
		device_ptr<cfloat> dst_ptr, feed& f)
{
	const char* kernel_string = R"aura_kernel(
		#include <boost/aura/backend.hpp>
		AURA_KERNEL
		void norm2_cfloat(AURA_GLOBAL cfloat* src_ptr, 
				AURA_GLOBAL cfloat* dst_ptr) {} 
	)aura_kernel";
	backend::kernel k = src_ptr.get_device().load_from_string(
			kernel_string, "norm2_cfloat",
			AURA_BACKEND_COMPILE_FLAGS);
	invoke(k, size, args(src_ptr.get(), dst_ptr.get()), f);
}

}

template <typename DeviceRangeType>
void norm2(DeviceRangeType& input_range, 
		DeviceRangeType& output_range, feed& f)
{
	detail::norm2(aura::traits::begin(input_range),
			aura::traits::size(input_range),
			aura::traits::begin(output_range), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_NORM2_HPP

