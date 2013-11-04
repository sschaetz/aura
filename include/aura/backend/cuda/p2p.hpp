#ifndef AURA_BACKEND_CUDA_P2P_HPP
#define AURA_BACKEND_CUDA_P2P_HPP

#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

inline void enable_peer_access(device & d1, device & d2) {
  // enable access from 1 to 2
  d1.set();
  AURA_CUDA_SAFE_CALL(cuCtxEnablePeerAccess(d2.get_backend_context(), 0));
  d1.unset();
  // enable access from 2 to 1
  d2.set();
  AURA_CUDA_SAFE_CALL(cuCtxEnablePeerAccess(d1.get_backend_context(), 0));
  d2.unset();
}

} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_P2P_HPP

