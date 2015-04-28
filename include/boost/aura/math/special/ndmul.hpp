#ifndef AURA_MATH_SPECIAL_NDMUL_HPP
#define AURA_MATH_SPECIAL_NDMUL_HPP

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

inline std::tuple<const char*,const char*> get_ndmul_kernel(float, float, float)
{
	return std::make_tuple("mul_float",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void mul_float(AURA_GLOBAL float* src,
			AURA_GLOBAL float* src_nd,
			AURA_GLOBAL float* dst_nd,
			unsigned long n,
			unsigned long N_nd)
	{
		unsigned int i = get_mesh_id();
		if (i < N_nd) {
			dst_nd[i] = src[i%n] * src_nd[i];
		}		
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_ndmul_kernel(float, cfloat, cfloat)
{
    return std::make_tuple("mul_float",
            R"aura_kernel(

    #include <boost/aura/backend.hpp>

    AURA_KERNEL void mul_float(AURA_GLOBAL float* src,
            AURA_GLOBAL cfloat* src_nd,
            AURA_GLOBAL cfloat* dst_nd,
            unsigned long n,
            unsigned long N_nd)
    {
        unsigned int i = get_mesh_id();
        if (i < N_nd) {
            dst_nd[i] = cmulf(make_cfloat(src[i%n],0), src_nd[i]);
        }
    }

        )aura_kernel");
}

inline std::tuple<const char*,const char*> get_ndmul_kernel(cfloat, cfloat, cfloat)
{
	return std::make_tuple("mul_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void mul_cfloat(AURA_GLOBAL cfloat* src,
			AURA_GLOBAL cfloat* src_nd,
			AURA_GLOBAL cfloat* dst_nd,
			unsigned long n,
			unsigned long N_nd)
	{
		unsigned int i = get_mesh_id();
		if (i < N_nd) {
			dst_nd[i] = cmulf(src[i%n], src_nd[i]);
		}		
	}
		
		)aura_kernel");	
}



} // namespace detail


template <typename DeviceRangeType1, typename DeviceRangeType2, typename DeviceRangeType3>
void ndmul(const DeviceRangeType1& input_range1,
		const DeviceRangeType2& input_range2,
		DeviceRangeType3& output_range,
		feed& f)
{	
    // assert that the output vector has the size of the first input vector
    assert(aura::traits::size(input_range2) ==
			aura::traits::size(output_range));
	// assert that the first input vector has the multiple size of the second input vector
	assert( (aura::traits::size(input_range2) %
			aura::traits::size(input_range1)) == 0);
    // assert that vectors life on the same device
	assert(aura::traits::get_device(input_range1) == 
			aura::traits::get_device(input_range2));
	assert(aura::traits::get_device(input_range1) == 
			aura::traits::get_device(output_range));        
    // deactivate these asserts by defining NDEBUG

    std::tuple<const char*, const char*> kernel_data;
    
	// matrix matrix multiplication
	kernel_data = detail::get_ndmul_kernel(
			aura::traits::get_value_type(input_range1),
			aura::traits::get_value_type(input_range2),
			aura::traits::get_value_type(output_range));

	backend::kernel k = aura::traits::get_device(output_range).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS, true);

	// input_range2 > input_range1 
	invoke(k, aura::traits::bounds(input_range2), 
			args(aura::traits::data(input_range1), 
				aura::traits::data(input_range2),
				aura::traits::data(output_range),
				aura::traits::size(input_range1),
				aura::traits::size(input_range2)), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_SPECIAL_NDMUL_HPP

