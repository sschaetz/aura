#ifndef AURA_MATH_MEMSET_ONES_HPP
#define AURA_MATH_MEMSET_ONES_HPP

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

inline std::tuple<const char*,const char*> memset_ones_kernel_name(
                float dst_ptr)
{
        return std::make_tuple("memset_ones_float",
            R"aura_kernel(
	
	#include <boost/aura/backend.hpp>

        AURA_KERNEL void memset_ones_float(AURA_GLOBAL float* dst_ptr)
	{
            dst_ptr[get_mesh_id()] = 1.; // the most simple kernel I wrote yet
	}
		
		)aura_kernel");	
}

inline std::tuple<const char*,const char*> memset_ones_kernel_name(
                cfloat dst_ptr)
{
        return std::make_tuple("memset_ones_cfloat",
            R"aura_kernel(

        #include <boost/aura/backend.hpp>

        AURA_KERNEL void memset_ones_cfloat(AURA_GLOBAL cfloat* dst_ptr)
        {
            unsigned long id = get_mesh_id();
                dst_ptr[id] = make_cfloat(1.,0.);
        }

                )aura_kernel");
}


} // namespace detail

template <typename DeviceRangeType>
void memset_ones(DeviceRangeType& output_range, feed& f)
{
        // get suitable kernel string for the ouput value type
        auto kernel_data = detail::memset_ones_kernel_name(
                        aura::traits::get_value_type(output_range));

        // compile kernel
        backend::kernel k = aura::traits::get_device(output_range).
                load_from_string(std::get<0>(kernel_data),
                                std::get<1>(kernel_data),
                                AURA_BACKEND_COMPILE_FLAGS);

        // define mesh size
        std::size_t N = aura::traits::size(output_range); // number of elements in output_range
        aura::mesh me = aura::mesh(N);                  // allocate the mesh

        // run kernel
        invoke(k, me, args(aura::traits::begin_raw(output_range)),f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_MEMSET_ONES_HPP

