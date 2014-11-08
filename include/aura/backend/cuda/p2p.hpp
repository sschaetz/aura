#ifndef AURA_BACKEND_CUDA_P2P_HPP
#define AURA_BACKEND_CUDA_P2P_HPP

#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

inline void enable_peer_access(device & d1, device & d2) 
{
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

inline std::vector<int> get_peer_access_matrix() 
{
	int ngpus = device_get_count();
	std::vector<int> r(ngpus*ngpus, 0);
	std::vector<device> gpus;
	for (int i=0; i<ngpus; i++) {
		gpus.push_back(device(i));	
	}
	for (auto& gpu1 : gpus) {
		for (auto& gpu2 : gpus) {
			int gpu1o = gpu1.get_ordinal();
			int gpu2o = gpu2.get_ordinal();
			if (gpu1o == gpu2o) {
				continue;	
			}
			AURA_CUDA_SAFE_CALL(cuDeviceCanAccessPeer(
						&r[gpu1o*ngpus+gpu2o],
						gpu1.get_backend_device(),
						gpu2.get_backend_device()));
		}
	}

	return r;
}

} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_P2P_HPP

