#ifndef AURA_MATH_BASIC_DIV_HPP
#define AURA_MATH_BASIC_DIV_HPP

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

inline std::tuple<const char*,const char*> get_div_kernel(float, float, float)
{
	return std::make_tuple("div_float",
			R"aura_kernel( 
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void div_float(AURA_GLOBAL float* src1,
			AURA_GLOBAL float* src2,
			AURA_GLOBAL float* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = src1[i] / src2[i];
		}		
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_div_kernel(cfloat, cfloat, cfloat)
{
	return std::make_tuple("div_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>
	
	AURA_KERNEL void div_cfloat(AURA_GLOBAL cfloat* src1,
			AURA_GLOBAL cfloat* src2,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = cdivf(src1[i], src2[i]);
		}				
	}
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_div_kernel(float, cfloat, cfloat)
{
	return std::make_tuple("add_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void add_cfloat(AURA_GLOBAL float* src1,
			AURA_GLOBAL cfloat* src2,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {								
			// The following implementation is taken from the LLVM Compiler Infrastructure, licensed under 
			// the MIT and the University of Illinois Open Source Licenses.
			// https://code.openhub.net/file?fid=XBgmMXzw1oxpd_pKEX4Olpef3gM&cid=DwH1iTUyTao&s=__divsc3&fp=406477&mp&projSelected=true#L0
	
			float a = src1[i];			
			float c = src2[i].x;
			float d = src2[i].y;
	
			int ilogbw = 0;
			float logbw = logb(fmax(fabs(c), fabs(d)));			
			ilogbw = (int)logbw;
			c = ldexp(c, -ilogbw);
			d = ldexp(d, -ilogbw);
			
			float denom = 1.0f / (c * c + d * d);
			float re = ldexp((a * c ) * denom, -ilogbw);
			float im = ldexp((- a * d) * denom, -ilogbw);
			dst[i] = make_cfloat(re, im);				
				}		
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> get_div_kernel(cfloat, float, cfloat)
{
	return std::make_tuple("add_cfloat",
			R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void add_cfloat(AURA_GLOBAL cfloat* src1,
			AURA_GLOBAL float* src2,
			AURA_GLOBAL cfloat* dst,
			unsigned long N)
	{
		unsigned int i = get_mesh_id();
		if (i < N) {
			dst[i] = make_cfloat(src1[i].x/src2[i], src1[i].y/src2[i]);
			
		}		
	}
		
		)aura_kernel");	
}

// kernels to divide matrix with scalar

inline std::tuple<const char*,const char*> get_scal_div_kernel(float, float, float)
{
    return std::make_tuple("scal_div_float",
            R"aura_kernel(

    #include <boost/aura/backend.hpp>

    AURA_KERNEL void scal_div_float(AURA_GLOBAL float* src1,
            AURA_GLOBAL float* src2,
            AURA_GLOBAL float* dst,
            unsigned long N)
    {
        unsigned int i = get_mesh_id();
        if (i < N) {
            dst[i] = src1[i]/src2[0];
        }
    }

        )aura_kernel");
}

inline std::tuple<const char*,const char*> get_scal_div_kernel(cfloat, cfloat, cfloat)
{
    return std::make_tuple("scal_div_cfloat",
            R"aura_kernel(

    #include <boost/aura/backend.hpp>

    AURA_KERNEL void scal_div_cfloat(AURA_GLOBAL cfloat* src1,
            AURA_GLOBAL cfloat* src2,
            AURA_GLOBAL cfloat* dst,
            unsigned long N)
    {
        unsigned int i = get_mesh_id();
        if (i < N) {
            dst[i] = cdivf(src1[i], src2[0]);
        }
    }

        )aura_kernel");
}


inline std::tuple<const char*,const char*> get_scal_div_kernel(float, cfloat, cfloat)
{
    return std::make_tuple("scal_div_cfloat",
            R"aura_kernel(

    #include <boost/aura/backend.hpp>

    AURA_KERNEL void scal_div_cfloat(AURA_GLOBAL float* src1,
            AURA_GLOBAL cfloat* src2,
            AURA_GLOBAL cfloat* dst,
            unsigned long N)
    {
        unsigned int i = get_mesh_id();
        if (i < N) {
            // The following implementation is taken from the LLVM Compiler Infrastructure, licensed under 
			// the MIT and the University of Illinois Open Source Licenses.
			// https://code.openhub.net/file?fid=XBgmMXzw1oxpd_pKEX4Olpef3gM&cid=DwH1iTUyTao&s=__divsc3&fp=406477&mp&projSelected=true#L0
	
			float a = src1[i];			
			float c = src2[0].x;
			float d = src2[0].y;
	
			int ilogbw = 0;
			float logbw = logb(fmax(fabs(c), fabs(d)));			
			ilogbw = (int)logbw;
			c = ldexp(c, -ilogbw);
			d = ldexp(d, -ilogbw);
			
			float denom = 1.0f / (c * c + d * d);
			float re = ldexp((a * c ) * denom, -ilogbw);
			float im = ldexp((- a * d) * denom, -ilogbw);
			dst[i] = make_cfloat(re, im);
        }
    }

        )aura_kernel");
}

inline std::tuple<const char*,const char*> get_scal_div_kernel(cfloat, float, cfloat)
{
    return std::make_tuple("scal_div_cfloat",
            R"aura_kernel(

    #include <boost/aura/backend.hpp>

    AURA_KERNEL void scal_div_cfloat(AURA_GLOBAL cfloat* src1,
            AURA_GLOBAL float* src2,
            AURA_GLOBAL cfloat* dst,
            unsigned long N)
    {
        unsigned int i = get_mesh_id();
        if (i < N) {
            dst[i] = make_cfloat(src1[i].x/src2[0], src1[i].y/src2[0]);
        }
    }

        )aura_kernel");
}

} // namespace detail




template <typename DeviceRangeType1, typename DeviceRangeType2, typename DeviceRangeType3>
void div(const DeviceRangeType1& input_range1,
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

    // check whether to use matrix matrix division or matrix scalar division
    std::tuple<const char*, const char*> kernel_data;   
    if  (aura::traits::size(input_range1) == aura::traits::size(input_range2))
    {
        // matrix matrix multiplication
        kernel_data = detail::get_div_kernel(
                aura::traits::get_value_type(input_range1),
                aura::traits::get_value_type(input_range2),
                aura::traits::get_value_type(output_range));
    }
    else
    {
        // if the dimensions not equal, assert that the second argument is a scalar
        assert(aura::traits::size(input_range2) == 1);

        // get kernel for matrix scalar multiplication
        kernel_data = detail::get_scal_div_kernel(
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

#endif // AURA_MATH_BASIC_DIV_HPP

