#ifndef AURA_MATH_BLAS_SUM_HPP
#define AURA_MATH_BLAS_SUM_HPP
#define BUNDLE_SIZE 128
// warning: BUNDLE_SIZE has to also be defined within the kernels

#include <tuple>

#include <boost/aura/meta/traits.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/math/complex.hpp>

// sum function specific includes
#include <boost/aura/math/memset_zero.hpp>
#include <boost/aura/math/partition_mesh.hpp>


// TODO: solve ambiguity of type (type changes by user) through wrapper function
// just like conj()

namespace boost
{
namespace aura
{
namespace math
{

namespace detail
{

inline std::tuple<const char*,const char*> sum_kernel_name(
                float src_ptr)
{
    return std::make_tuple("sum_float",
            R"aura_kernel(
	
	#include <boost/aura/backend.hpp>
    #define BUNDLE_SIZE 128

    AURA_KERNEL void sum_float(AURA_GLOBAL float* src_ptr,
                        AURA_GLOBAL float* dst_ptr, unsigned long N)  // are there better ways to pass the vector size ??
	{
            // get bundle and mesh id
            unsigned int bid = get_bundle_id(); // goes from 0 ... bundle_size-1
            unsigned int id  = get_mesh_id();   // goes from 0 ... numel(A)-1 ?

            // allocate shared memory
            AURA_SHARED float sm[BUNDLE_SIZE];

            // copy an element od the input vector to the shared memory block of this bundle
            if (id >= N)
                // deal with mesh_ids that are greater than the actual vector size
                sm[bid] = 0;
            else
                // copy the sdata
                sm[bid] = src_ptr[id];

            // wait until all fibers within the bundle are done, i.e. all 16 elements of the shared memory are filled
            AURA_SYNC

            if (bid < 64) // divide the sm in 2 blocks and add the second block to the first
               sm[bid] += sm[bid+64];
            AURA_SYNC

            if (bid < 32) // again
              sm[bid] += sm[bid+32];
            AURA_SYNC

            if (bid < 16) // again
              sm[bid] += sm[bid+16];
            AURA_SYNC

            if (bid < 8) // again
             sm[bid] += sm[bid+8];
            AURA_SYNC

            if (bid < 4) // again
               sm[bid] += sm[bid+4];
            AURA_SYNC

            if (bid < 2) // again
               sm[bid] += sm[bid+2];
            AURA_SYNC

            if (bid < 1) {
               sm[0] += sm[1];
               atomic_addf(dst_ptr, sm[0]); // accumulate result but don't allow for multiple access of the memory position (therefor atomic)
            }
	}
		
		)aura_kernel");	
}


inline std::tuple<const char*,const char*> sum_kernel_name(
                cfloat src_ptr)
{
        return std::make_tuple("sum_cfloat",
            R"aura_kernel(

        #include <boost/aura/backend.hpp>
        #define BUNDLE_SIZE 128

        AURA_KERNEL void sum_cfloat(AURA_GLOBAL cfloat* src_ptr,
                        AURA_GLOBAL cfloat* dst_ptr, unsigned long N)
        {
            // get bundle and mesh id
            unsigned int bid = get_bundle_id(); // goes from 0 ... bundle_size-1
            unsigned int id  = get_mesh_id();   // goes from 0 ... numel(A)-1 ?

            // allocate shared memory
            AURA_SHARED cfloat sm[BUNDLE_SIZE];

            // copy an element od the input vector to the shared memory block of this bundle
            if (id >= N){
                // deal with mesh_ids that are greater than the actual vector size
                sm[bid] = make_cfloat(0,0);
             }
            else{
                // copy the squared magnitude
                sm[bid] = src_ptr[id];
             }

            // wait until all fibers within the bundle are done, i.e. all 16 elements of the shared memory are filled
            AURA_SYNC

            if (bid < 64) // divide the sm in 2 blocks and add the second block to the first
               sm[bid] += sm[bid+64];
            AURA_SYNC

            if (bid < 32) // again
              sm[bid] += sm[bid+32];
            AURA_SYNC

            if (bid < 16) // again
              sm[bid] += sm[bid+16];
            AURA_SYNC

            if (bid < 8) // again
             sm[bid] += sm[bid+8];
            AURA_SYNC

            if (bid < 4) // again
               sm[bid] += sm[bid+4];
            AURA_SYNC

            if (bid < 2) // again
               sm[bid] += sm[bid+2];
            AURA_SYNC

            if (bid < 1) {                
               sm[0] += sm[1];
               atomic_addf(dst_ptr, sm[0]); // accumulate result but don't allow for multiple access of the memory position (therefor atomic)
            }
        }

                )aura_kernel");
}


}



template <typename DeviceRangeType1, typename DeviceRangeType2>
void sum(const DeviceRangeType1& input_range,
                DeviceRangeType2& output_range, feed& f)
{
        //---- preprocessing ---

        // set output_range to zero
        aura::math::memset_zero(output_range, f);


        //---- summation ---

        // get the correct kernel string according to the data type
        auto kernel_data = detail::sum_kernel_name(
                        aura::traits::get_value_type(input_range));

        // std::cout << "kernel_data: " << kernel_data << std::endl;

        // compile kernel
        backend::kernel k = aura::traits::get_device(output_range).
                load_from_string(std::get<0>(kernel_data),
                                std::get<1>(kernel_data),
                                AURA_BACKEND_COMPILE_FLAGS);


        // define mesh and bundle size
        std::size_t trueN = aura::traits::size(input_range);// number of elements in input_range...
        std::size_t N = round2bs(trueN);                    // ...rounded to a multiple of BUNDLE_SIZE
        std::vector<std::size_t> mesh_size(3,1);

        // partition the mesh size
        partition_mesh(mesh_size,N,BUNDLE_SIZE);

        #ifdef DEBUG
        // debug output (mesh size and mesh overhead)
        std::cout   << "mesh_size = " << mesh_size[0] << " " << mesh_size[1] << " " << mesh_size[2] << std::endl
                    << " total necessary " << N  << std::endl
                    << " total activated " << mesh_size[0]*mesh_size[1]*mesh_size[2] << std::endl;
        #endif

        // allocate the mesh
        aura::mesh me = aura::mesh(mesh_size[0],mesh_size[1],mesh_size[2]);
        aura::bundle bun = bundle(BUNDLE_SIZE);     // set the bundle size

        // run kernel
        invoke(k, me, bun, args(aura::traits::begin_raw(input_range),
                        aura::traits::begin_raw(output_range), trueN),f);


        return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_BLAS_SUM_HPP

