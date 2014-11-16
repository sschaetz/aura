#ifndef AURA_BACKEND_CUDA_INIT_HPP
#define AURA_BACKEND_CUDA_INIT_HPP

#include <cuda.h>
#include <boost/aura/backend/cuda/call.hpp>
#include <boost/aura/misc/deprecate.hpp>

namespace boost
{
namespace aura {
namespace backend_detail {
namespace cuda {

inline void init() {
  AURA_CUDA_SAFE_CALL(cuInit(0));
}

/// initialize backend
inline void initialize() {
  AURA_CUDA_SAFE_CALL(cuInit(0));
}

DEPRECATED(void init());

} // cuda
} // backend_detail
} // aura
} // boost

#endif // AURA_BACKEND_CUDA_INIT_HPP

