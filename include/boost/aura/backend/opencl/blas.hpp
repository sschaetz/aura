#ifndef AURA_BACKEND_OPENCL_BLAS_HPP
#define AURA_BACKEND_OPENCL_BLAS_HPP

#include <tuple>
#include <boost/move/move.hpp>
#include <boost/aura/backend/opencl/call.hpp>
#include <boost/aura/backend/opencl/device_ptr.hpp>
#include <boost/aura/backend/opencl/memory.hpp>
#include <boost/aura/detail/svec.hpp>
#include <boost/aura/backend/opencl/device.hpp>
#include <boost/aura/bounds.hpp>
#include <clBLAS.h>


namespace boost
{
namespace aura
{
namespace backend_detail
{
namespace opencl
{

typedef std::size_t blas_size;
typedef svec<fft_size, 3> blas_embed;
using ::boost::aura::bounds;



inline void blas_initialize(void)
{
	AURA_CLBLAS_SAFE_CALL(clblasSetup());
}

inline void blas_terminate(void)
{
	clblasTeardown();
}

inline void gemv(const device_ptr<float>& A, const device_ptr<float>& x,
		const device_ptr<float>& y, bounds b, feed& f)
{
	size_t M = b[0];
	size_t N = b[1];

	int errorcode =
	clblasSgemv(clblasRowMajor, clblasNoTrans,
				M, N,
				1.0,
				A.get_base(), 0, N, x.get_base(), 0, 1,
				0.0,
				y.get_base(), 0, 1, 1,
				&f.get_backend_stream(), 0, NULL, NULL);
	AURA_OPENCL_CHECK_ERROR(errorcode);

}


} // opencl
} // backend_detail
} // aura
} // boost

#endif // AURA_BACKEND_OPENCL_BLAS_HPP

