#ifndef AURA_BLAS_HPP
#define AURA_BLAS_HPP

#include <boost/aura/backend.hpp>

namespace boost {
namespace aura {

using backend::blas_initialize;
using backend::blas_terminate;

using backend::gemv;

void gemv(const device_array<float>& A, const device_array<float>& x, 
		device_array<float>& y, feed& f) 
{
	gemv(A.begin(), x.begin(), y.begin(), A.get_bounds(), f);
}


} // aura
} // boost 

#endif // AURA_BLAS_HPP

