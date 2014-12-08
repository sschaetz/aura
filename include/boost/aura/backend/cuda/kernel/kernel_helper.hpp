#ifndef AURA_BACKEND_CUDA_KERNEL_HELPER_HPP
#define AURA_BACKEND_CUDA_KERNEL_HELPER_HPP

#include <cuda.h>

__device__ __forceinline__ unsigned int get_mesh_id() {
	return (gridDim.y*gridDim.x*blockIdx.z + 
			gridDim.x*blockIdx.y + blockIdx.x) * 
		(blockDim.z*blockDim.y*blockDim.x) + 
		blockDim.y*blockDim.x*threadIdx.z + blockDim.x*threadIdx.y + 
		threadIdx.x;
}

__device__ __forceinline__ unsigned int get_mesh_size() {
	return gridDim.z*gridDim.y*gridDim.x * blockDim.z*blockDim.y*blockDim.x;
}

__device__ __forceinline__ unsigned int get_bundle_id() {
	return blockDim.y*blockDim.x*threadIdx.z + 
		blockDim.x*threadIdx.y + threadIdx.x;
}

#define AURA_KERNEL extern "C" __global__
#define AURA_GLOBAL 
#define AURA_DEVICE_FUNCTION __device__

#endif // AURA_BACKEND_CUDA_KERNEL_HELPER_HPP

