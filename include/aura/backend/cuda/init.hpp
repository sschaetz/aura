#ifdef AURA_BACKEND_CUDA_INIT_HPP
#define AURA_BACKEND_CUDA_INIT_HPP

#include <cuda.h>
#include <aura/backend/cuda/call.hpp>

namespace aura {
namespace backend {
namespace cuda {

inline void init() {
  AURA_CUDA_SAFE_CALL(cuInit(0));
}

} // cuda
} // backend
} // aura

#endif // AURA_BACKEND_CUDA_INIT_HPP

