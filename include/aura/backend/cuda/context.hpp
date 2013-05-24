#ifndef AURA_BACKEND_CUDA_CONTEXT_HPP
#define AURA_BACKEND_CUDA_CONTEXT_HPP

#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>

namespace aura {
namespace backend {
namespace cuda {

/// context handle
typedef CUcontext context;

/**
 * create context for device 
 *
 * @param d the device handle 
 * @return the context handle
 */
inline context context_create(device d) {
  context c;
  AURA_CUDA_SAFE_CALL(cuCtxCreate(&c, 0, d));
  return c;
}

/**
 * destroy context
 */
inline void context_destroy(context c) {
  AURA_CUDA_SAFE_CALL(cuCtxDestroy(c));
}

} // cuda
} // backend
} // aura



#endif // AURA_BACKEND_CUDA_CONTEXT_HPP

