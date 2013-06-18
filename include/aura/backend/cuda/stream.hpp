#ifndef AURA_BACKEND_CUDA_STREAM_HPP
#define AURA_BACKEND_CUDA_STREAM_HPP


#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

/// device handle
typedef CUstream stream;

/**
 * create stream for device context
 *
 * @param device device to create stream for
 * @param context context to create stream for (must match device)
 * @return the stream handle
 */
inline stream stream_create(device d, context c) {
  stream s; 
  AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(c));
  AURA_CUDA_SAFE_CALL(cuStreamCreate(&s, 0/*, CU_STREAM_NON_BLOCKING*/));
  AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(NULL));
  return s;
}

/**
 * create default stream for device context
 *
 * @param device device to create stream for
 * @param context context to create stream for (must match device)
 * @return the default stream handle
 */
inline stream stream_create_default(device d, context c) {
  return 0; // the CUDA default stream is simply 0
}

/**
 * wait (block) until all operations in the stream are finished
 *
 * @param s stream
 */
inline void stream_synchronize(stream s) {
  AURA_CUDA_SAFE_CALL(cuStreamSynchronize(s));
}

/**
 * destroy stream
 *
 * @param stream stream that should be destroyed
 */
inline void stream_destroy(stream s) {
  AURA_CUDA_SAFE_CALL(cuStreamDestroy(s));
}

} // cuda
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_STREAM_HPP

