#ifndef AURA_MATH_BLAS_DOT_HPP
#define AURA_MATH_BLAS_DOT_HPP

#include <tuple>
#include <cassert>

#include <stdio.h>
#include <boost/aura/meta/traits.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/math/complex.hpp>
#include <boost/aura/math/memset_zero.hpp>
#include <boost/aura/math/partition_mesh.hpp>

namespace boost
{
namespace aura
{
namespace math
{

namespace detail
{

inline std::tuple<const char*,const char*> get_dot_kernel(float, float, float)
{
	return std::make_tuple("dot_float",
			R"aura_kernel(

	#include <boost/aura/backend.hpp>

	AURA_KERNEL void dot_float(AURA_GLOBAL float* src1,
			AURA_GLOBAL float* src2,
			AURA_GLOBAL float* dst,
			const unsigned long N)
	{
		unsigned int mid = get_mesh_id();
		unsigned int bid = get_bundle_id();

		AURA_SHARED float sm[256];

		if ( mid < N ) {
			sm[bid] 	= src1[mid]*src2[mid];

		} else {
			sm[bid] = 0;
			//return;
		}

		AURA_SYNC;

		if (bid < 128)
			sm[bid] += sm[bid+128];
		AURA_SYNC;

		if (bid < 64)
			sm[bid] += sm[bid+64];
		AURA_SYNC;

		if (bid < 32)
			sm[bid] += sm[bid+32];
		AURA_SYNC;

		if (bid < 16)
			sm[bid] += sm[bid+16];
		AURA_SYNC;

		if (bid < 8)
			sm[bid] += sm[bid+8];
		AURA_SYNC;

		if (bid < 4)
			sm[bid] += sm[bid+4];
		AURA_SYNC;

		if (bid < 2)
			sm[bid] += sm[bid+2];
		AURA_SYNC;

		if (bid < 1) {
			sm[bid] += sm[bid+1];
			atomic_addf(dst, sm[0]);
		}
		AURA_SYNC;
	}

		)aura_kernel");
}

inline std::tuple<const char*,const char*> get_dot_kernel(cfloat, cfloat, cfloat)
{
	return std::make_tuple("dot_cfloat",
			R"aura_kernel(

	#include <boost/aura/backend.hpp>

	AURA_KERNEL void dot_cfloat(AURA_GLOBAL cfloat* src1,
			AURA_GLOBAL cfloat* src2,
			AURA_GLOBAL cfloat* dst,
			const unsigned long N)
	{
		unsigned int mid = get_mesh_id();
		unsigned int bid = get_bundle_id();
		AURA_SHARED cfloat sm[256];

		if ( mid < N ) {
            sm[bid] 	= cmulf(conjf(src1[mid]),src2[mid]);
		} else {
            sm[bid] = make_cfloat(0.0,0.0);
		}
		AURA_SYNC;

		if (bid < 128)
			sm[bid] += sm[bid+128];
		AURA_SYNC;

		if (bid < 64)
			sm[bid] += sm[bid+64];
		AURA_SYNC;

		if (bid < 32)
			sm[bid] += sm[bid+32];
		AURA_SYNC;

		if (bid < 16)
			sm[bid] += sm[bid+16];
		AURA_SYNC;

		if (bid < 8)
			sm[bid] += sm[bid+8];
		AURA_SYNC;

		if (bid < 4)
			sm[bid] += sm[bid+4];
		AURA_SYNC;

		if (bid < 2)
			sm[bid] += sm[bid+2];
		AURA_SYNC;

		if (bid < 1) {
			sm[bid] += sm[bid+1];
 			atomic_addf(crealfp(dst), crealf(sm[0]));
            if (src1 != src2)
                atomic_addf(cimagfp(dst), cimagf(sm[0]));
		}
		AURA_SYNC;
	}

		)aura_kernel");
}

} // namespace detail



template <typename DeviceRangeType>
void dot(const DeviceRangeType& input_range1,
		const DeviceRangeType& input_range2,
		DeviceRangeType& output_range, feed& f)
{
	// asserts to make sure vectors have same size
	assert(aura::traits::size(input_range1) ==
			aura::traits::size(input_range2));
	assert(aura::traits::size(output_range) == 1 );
	// and vectors life on the same device
	assert(aura::traits::get_device(input_range1) ==
			aura::traits::get_device(input_range2));
	assert(aura::traits::get_device(input_range1) ==
			aura::traits::get_device(output_range));
	// deactivate these asserts by defining NDEBUG



	auto kernel_data = detail::get_dot_kernel(
			aura::traits::get_value_type(input_range1),
			aura::traits::get_value_type(input_range2),
			aura::traits::get_value_type(output_range));

	backend::kernel k = aura::traits::get_device(output_range).
		load_from_string(std::get<0>(kernel_data),
				std::get<1>(kernel_data),
				AURA_BACKEND_COMPILE_FLAGS, true);

	long vs = aura::traits::size(input_range1);
	int bs = 256;
	std::vector<std::size_t> mesh_size(3,1);

	// partition the mesh size
	partition_mesh(mesh_size,vs,bs);


	// set output_range to zero
    aura::math::memset_zero(output_range, f);

	invoke(k, mesh(mesh_size[0], mesh_size[1], mesh_size[2]), bundle(bs),
			args(aura::traits::begin_raw(input_range1),
				aura::traits::begin_raw(input_range2),
				aura::traits::begin_raw(output_range),
				aura::traits::size(input_range1)), f);
	return;
}

} // namespace math
} // namespace aura
} // namespace boost

#endif // AURA_MATH_BLAS_DOT_HPP

