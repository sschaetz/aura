#ifndef AURA_MATH_BASIC_SQRT_HPP
#define AURA_MATH_BASIC_SQRT_HPP

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

inline std::tuple<const char*,const char*> sqrt_kernel_name(float src_ptr, float dst_ptr)
{
        return std::make_tuple("sqrt_float",
            R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

        AURA_KERNEL void sqrt_float(AURA_GLOBAL float* src_ptr,
				AURA_GLOBAL float* dst_ptr)
	{
            unsigned int id  = get_mesh_id();
            dst_ptr[id] = sqrt(src_ptr[id]);
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> sqrt_kernel_name(cfloat src_ptr, cfloat dst_ptr)
{
        return std::make_tuple("sqrt_cfloat",
            R"aura_kernel(

        #include <boost/aura/backend.hpp>

        inline float sgn(float x){return x>=0? 1.:-1.;}

        AURA_KERNEL void sqrt_cfloat(AURA_GLOBAL cfloat* src_ptr,
				AURA_GLOBAL cfloat* dst_ptr)
        {

            unsigned int id  = get_mesh_id();

            // get complex shortcuts
            float re  = crealf(src_ptr[id]);
            float im  = cimagf(src_ptr[id]);
            float abs = cabsf(src_ptr[id]);

            // avoid sqrt(<0) due to rounding errors
            float secureDiff = (abs - re)/ 2;
            if (secureDiff < 0)
                secureDiff = 0;

            // complex sqrt
            dst_ptr[id] = make_cfloat(sqrt((abs + re)/2),
                                   sgn(im) * sqrt(secureDiff));

        }

                )aura_kernel");
}


}

template <typename DeviceRangeType>
void sqrt(const DeviceRangeType& input_range,
	  DeviceRangeType& output_range, feed& f)
{

	assert(aura::traits::get_device(input_range) ==
		aura::traits::get_device(output_range));


        auto kernel_data = detail::sqrt_kernel_name(
			aura::traits::get_value_type(input_range),
			aura::traits::get_value_type(output_range));

        // compile kernel
        backend::kernel k = aura::traits::get_device(output_range).
                load_from_string(std::get<0>(kernel_data),
                                std::get<1>(kernel_data),
                                AURA_BACKEND_COMPILE_FLAGS, true);

        // define mesh and bundle size
        std::size_t N = aura::traits::size(input_range); // number of elements in input_range
        aura::mesh me = aura::mesh(N);                    // allocate the mesh

        // run kernel
        invoke(k, me, args(aura::traits::begin_raw(input_range),
			aura::traits::begin_raw(output_range)),f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_BASIC_SQRT_HPP

