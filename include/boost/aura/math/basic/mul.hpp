#ifndef AURA_MATH_BASIC_MUL_HPP
#define AURA_MATH_BASIC_MUL_HPP

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

inline std::tuple<const char*,const char*> get_mul_kernel(float, float, float)
{
	return std::make_tuple("mul_float",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void mul_float(AURA_GLOBAL float* src1,
			AURA_GLOBAL float* src2,
			AURA_GLOBAL float* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = src1[i] * src2[i];
		}		
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_mul_kernel(cfloat, cfloat, cfloat)
{
	return std::make_tuple("mul_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void mul_cfloat(AURA_GLOBAL cfloat* src1,
			AURA_GLOBAL cfloat* src2,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = cmulf(src1[i], src2[i]);
		}		
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_mul_kernel(float, cfloat, cfloat)
{
	return std::make_tuple("mul_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void mul_cfloat(AURA_GLOBAL float* src1,
			AURA_GLOBAL cfloat* src2,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = make_cfloat(src1[i]*src2[i].x, src1[i]*src2[i].y);					
		}		
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_mul_kernel(cfloat, float, cfloat)
{
	return std::make_tuple("mul_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void mul_cfloat(AURA_GLOBAL cfloat* src1,
			AURA_GLOBAL float* src2,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = make_cfloat(src1[i].x*src2[i], src1[i].y*src2[i]);
		}		
	}
		
		)aura_kernel");	
}

// kernel to multiply matrix with scalar: so far, the scalar has to be the second argument in all cases

inline std::tuple<const char*,const char*> get_scal_mul_kernel(float, float, float)
{
    return std::make_tuple("scal_mul_float",
            R"aura_kernel(

    #include <boost/aura/backend.hpp>

    AURA_KERNEL void scal_mul_float(AURA_GLOBAL float* src1,
            AURA_GLOBAL float* src2,
            AURA_GLOBAL float* dst,
            unsigned long N)
    {
        unsigned int i = get_mesh_id();
        if (i < N) {
            dst[i] = src1[i] * src2[0];
        }
    }

        )aura_kernel");
}

inline std::tuple<const char*,const char*> get_scal_mul_kernel(cfloat, cfloat, cfloat)
{
    return std::make_tuple("scal_mul_cfloat",
            R"aura_kernel(

    #include <boost/aura/backend.hpp>

    AURA_KERNEL void scal_mul_cfloat(AURA_GLOBAL cfloat* src1,
            AURA_GLOBAL cfloat* src2,
            AURA_GLOBAL cfloat* dst,
            unsigned long N)
    {
        unsigned int i = get_mesh_id();
        if (i < N) {
            dst[i] = cmulf(src1[i], src2[0]);
        }
    }

        )aura_kernel");
}


inline std::tuple<const char*,const char*> get_scal_mul_kernel(float, cfloat, cfloat)
{
    return std::make_tuple("scal_mul_cfloat",
            R"aura_kernel(

    #include <boost/aura/backend.hpp>

    AURA_KERNEL void scal_mul_cfloat(AURA_GLOBAL float* src1,
            AURA_GLOBAL cfloat* src2,
            AURA_GLOBAL cfloat* dst,
            unsigned long N)
    {
        unsigned int i = get_mesh_id();
        if (i < N) {
            dst[i] = make_cfloat(src1[i]*src2[0].x, src1[i]*src2[0].y);
        }
    }

        )aura_kernel");
}

inline std::tuple<const char*,const char*> get_scal_mul_kernel(cfloat, float, cfloat)
{
    return std::make_tuple("scal_mul_cfloat",
            R"aura_kernel(

    #include <boost/aura/backend.hpp>

    AURA_KERNEL void scal_mul_cfloat(AURA_GLOBAL cfloat* src1,
            AURA_GLOBAL float* src2,
            AURA_GLOBAL cfloat* dst,
            unsigned long N)
    {
        unsigned int i = get_mesh_id();
        if (i < N) {
            dst[i] = make_cfloat(src1[i].x*src2[0], src1[i].y*src2[0]);
        }
    }

        )aura_kernel");
}


} // namespace detail


template <typename DeviceRangeType1, typename DeviceRangeType2, typename DeviceRangeType3>
void mul(const DeviceRangeType1& input_range1,
		const DeviceRangeType2& input_range2,
		DeviceRangeType3& output_range, feed& f)
{	
    // assert that the output vector has the size of the first input vector
    assert(aura::traits::size(input_range1) ==
			aura::traits::size(output_range));
    // assert that vectors life on the same device
	assert(aura::traits::get_device(input_range1) == 
			aura::traits::get_device(input_range2));
	assert(aura::traits::get_device(input_range1) == 
			aura::traits::get_device(output_range));        
    // deactivate these asserts by defining NDEBUG

    // check wether to use matrix matrix multipication or matrix scalar mltiplication
    std::tuple<const char*, const char*> kernel_data;
    if  (aura::traits::size(input_range1) == aura::traits::size(input_range2))
    {
        // matrix matrix multiplication
        kernel_data = detail::get_mul_kernel(
                aura::traits::get_value_type(input_range1),
                aura::traits::get_value_type(input_range2),
                aura::traits::get_value_type(output_range));
    }
    else
    {
        // if the dimensions not equal, assert that the second argument is a scalar
        assert(aura::traits::size(input_range2) == 1);

        // get kernel for matrix scalar multiplication
        kernel_data = detail::get_scal_mul_kernel(
                aura::traits::get_value_type(input_range1),
                aura::traits::get_value_type(input_range2),
                aura::traits::get_value_type(output_range));
    }


	backend::kernel k = aura::traits::get_device(output_range).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS, true);

	invoke(k, aura::traits::bounds(input_range1), 
			args(aura::traits::data(input_range1),
				aura::traits::data(input_range2),
				aura::traits::data(output_range),
				aura::traits::size(input_range1)), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_BASIC_MUL_HPP

