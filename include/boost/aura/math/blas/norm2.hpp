#ifndef AURA_MATH_BLAS_NORM2_HPP
#define AURA_MATH_BLAS_NORM2_HPP

#define AURA_MATH_BLAS_NORM2_BUNDLE_SIZE 128
// warning: BUNDLE_SIZE has to also be defined within the kernels

#include <tuple>

#include <boost/aura/meta/traits.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/math/complex.hpp>

// norm2 function specific includes
#include <boost/aura/math/memset_zero.hpp>
#include <boost/aura/math/basic/sqrt.hpp>
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

inline std::tuple<const char*,const char*> norm2_kernel_name(
                float src_ptr)
{
	return std::make_tuple("norm2_float",
            R"aura_kernel(
	
	#include <boost/aura/backend.hpp>
	#define AURA_MATH_BLAS_NORM2_BUNDLE_SIZE 128
	
	// are there better ways to pass the vector size ??
	AURA_KERNEL void norm2_float(AURA_GLOBAL float* src_ptr, 
                        AURA_GLOBAL float* dst_ptr, unsigned long N)  
	{
		// get bundle and mesh id
		
		// from 0 ... bundle_size-1
		unsigned int bid = get_bundle_id();
		// from 0 ... numel(A)-1 ?
		unsigned int id  = get_mesh_id();
		
		// allocate shared memory
		AURA_SHARED float sm[AURA_MATH_BLAS_NORM2_BUNDLE_SIZE];

		// copy an element od the input vector 
		// to the shared memory block of this bundle
		if (id >= N) {
			// deal with mesh_ids that are greater 
			// than the actual vector size
			sm[bid] = 0;
		} else {
			// copy the squared magnitude
			sm[bid] = src_ptr[id]*src_ptr[id];
		}
		// wait until all fibers within the bundle are done, 
		// i.e. all 16 elements of the shared memory are filled
		AURA_SYNC

		// divide the sm in 2 blocks 
		// and add the second block to the first
		if (bid < 64) {
			sm[bid] += sm[bid+64];
		}
		AURA_SYNC

		if (bid < 32)
			sm[bid] += sm[bid+32];
		AURA_SYNC

		if (bid < 16)
			sm[bid] += sm[bid+16];
		AURA_SYNC

		if (bid < 8)
			sm[bid] += sm[bid+8];
		AURA_SYNC

		if (bid < 4) 
			sm[bid] += sm[bid+4];
		AURA_SYNC

		if (bid < 2)
			sm[bid] += sm[bid+2];
		AURA_SYNC

		if (bid < 1) {
			sm[0] += sm[1];
			// accumulate result but don't allow for 
			// multiple access of the memory position 
			// (therefor atomic)
			atomic_addf(dst_ptr, sm[0]); 		
		}
	}
		
		)aura_kernel");	
}


inline std::tuple<const char*,const char*> norm2_kernel_name(
                cfloat src_ptr)
{
        return std::make_tuple("norm2_cfloat",
            R"aura_kernel(

        #include <boost/aura/backend.hpp>
        #define AURA_MATH_BLAS_NORM2_BUNDLE_SIZE 128

        AURA_KERNEL void norm2_cfloat(AURA_GLOBAL cfloat* src_ptr,
                        AURA_GLOBAL float* dst_ptr, unsigned long N)  // are there better ways to pass the vector size ??
        {
            // get bundle and mesh id
            unsigned int bid = get_bundle_id(); // goes from 0 ... bundle_size-1
            unsigned int id  = get_mesh_id();   // goes from 0 ... numel(A)-1 ?

            // allocate shared memory
            AURA_SHARED float sm[AURA_MATH_BLAS_NORM2_BUNDLE_SIZE];

            // copy an element od the input vector to the shared memory block of this bundle
            if (id >= N){
                // deal with mesh_ids that are greater than the actual vector size
                sm[bid] = 0.;
             }
            else{
                // copy the squared magnitude
                sm[bid] = crealf(src_ptr[id]) * crealf(src_ptr[id])
                                       + cimagf(src_ptr[id]) * cimagf(src_ptr[id]);
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

// "shortcut" inline function to round to the smallest multiple of bundle_size
template <typename T>
inline float round2bs(T x)
{
	return(ceil((float)x / AURA_MATH_BLAS_NORM2_BUNDLE_SIZE) * 
			AURA_MATH_BLAS_NORM2_BUNDLE_SIZE);
}


template <typename DeviceRangeType1, typename DeviceRangeType2>
void norm2(const DeviceRangeType1& input_range,
                DeviceRangeType2& output_range, feed& f)
{
        //---- preprocessing ---

        // set output_range to zero
        aura::math::memset_zero(output_range, f);




        //---- squared summation ---

        // get the correct kernel string according to the data type
        auto kernel_data = detail::norm2_kernel_name(
                        aura::traits::get_value_type(input_range));

        // compile kernel
        backend::kernel k = aura::traits::get_device(output_range).
                load_from_string(std::get<0>(kernel_data),
                                std::get<1>(kernel_data),
                                AURA_BACKEND_COMPILE_FLAGS);


        // define mesh and bundle size
	// number of elements in input_range
        std::size_t trueN = aura::traits::size(input_range);
	// rounded to a multiple of BUNDLE_SIZE
	std::size_t N = round2bs(trueN);                    
        std::vector<std::size_t> mesh_size(3,1);

        // partition the mesh size
        partition_mesh(mesh_size,N,AURA_MATH_BLAS_NORM2_BUNDLE_SIZE);

#if 0
        // debug output (mesh size and mesh overhead)
        std::cout  << "mesh_size = " << mesh_size[0] << 
		" " << mesh_size[1] << " " << mesh_size[2] << std::endl << 
		" total necessary " << N  << std::endl << 
		" total activated " << 
		mesh_size[0]*mesh_size[1]*mesh_size[2] << std::endl;
#endif
        // allocate the mesh
        aura::mesh me = aura::mesh(mesh_size[0],mesh_size[1],mesh_size[2]);
	// set the bundle size
        aura::bundle bun = bundle(AURA_MATH_BLAS_NORM2_BUNDLE_SIZE);     

        // run kernel
        invoke(k, me, bun, args(aura::traits::data(input_range),
                        aura::traits::data(output_range), trueN),f);



        //---- postprocessing ---
        aura::math::sqrt(output_range, output_range, f);


        return;
}

} // namespace math
} // namespace aura
} // namespace boost

#undef AURA_MATH_BLAS_NORM2_BUNDLE_SIZE

#endif // AURA_MATH_NORM2_HPP

