#ifndef AURA_MATH_ADD_KERNEL_HPP
#define AURA_MATH_ADD_KERNEL_HPP

#include <tuple>

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

inline std::tuple<const char*,const char*> get_add_kernel(float, float, float)
{
	return std::make_tuple("norm2_float",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void norm2_float(AURA_GLOBAL float* src1,
			AURA_GLOBAL float* src2,
			AURA_GLOBAL float* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = src1[i] + src2[i];
		}		
	}
		
		)aura_kernel");	
}

} // namespace detail
} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_ADD_KERNEL_HPP

