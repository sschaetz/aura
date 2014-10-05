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
  CUresult result = cuCtxEnablePeerAccess(d2.get_backend_context(), 0);
  if (result != CUDA_SUCCESS && 
		  result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
	  AURA_CUDA_CHECK_ERROR(result);
  }
  d1.unset();
  // enable access from 2 to 1
  d2.set();
  result = cuCtxEnablePeerAccess(d1.get_backend_context(), 0);
  if (result != CUDA_SUCCESS && 
		  result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
	  AURA_CUDA_CHECK_ERROR(result);
  }
  d2.unset();
}

} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_P2P_HPP

