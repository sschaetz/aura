#ifndef AURA_BACKEND_CUDA_INIT_HPP
#define AURA_BACKEND_CUDA_INIT_HPP

#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/misc/deprecate.hpp>

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

#endif // AURA_BACKEND_CUDA_INIT_HPP

