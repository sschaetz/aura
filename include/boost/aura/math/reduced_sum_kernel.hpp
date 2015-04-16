
#ifndef AURA_MATH_REDUCED_SUM_KERNEL_HPP
#define AURA_MATH_REDUCED_SUM_KERNEL_HPP

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
                for (int j = 0; j < dimensions; j++) {
                  if (i < n)
                       dst[i] += src[i+j*n] ; 
                }
	}
		
		)aura_kernel");	
}
} // namespace detail
} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_REDUCED_SUM_HPP
