#ifndef AURA_BACKEND_CUDA_KERNEL_HELPER_HPP
#define AURA_BACKEND_CUDA_KERNEL_HELPER_HPP

#include <cuda.h>

__device__ __forceinline__ unsigned int get_mesh_id() 
{
	return (gridDim.y*gridDim.x*blockIdx.z + 
			gridDim.x*blockIdx.y + blockIdx.x) * 
		(blockDim.z*blockDim.y*blockDim.x) + 
		blockDim.y*blockDim.x*threadIdx.z + blockDim.x*threadIdx.y + 
		threadIdx.x;
}

__device__ __forceinline__ unsigned int get_id_in_mesh() 
{
	return get_mesh_id();
}


__device__ __forceinline__ unsigned int get_mesh_size() 
{
	return gridDim.z*gridDim.y*gridDim.x * blockDim.z*blockDim.y*blockDim.x;
}

__device__ __forceinline__ unsigned int get_bundle_id() 
{
	return blockDim.y*blockDim.x*threadIdx.z + 
		blockDim.x*threadIdx.y + threadIdx.x;
}

__device__ __forceinline__ unsigned int get_id_in_bundle() 
{
	return get_bundle_id();
}

__device__ __forceinline__ unsigned int get_bundle_size() 
{
	return blockDim.y*blockDim.x*blockDim.z;
}


// mesh ids
__device__ __forceinline__ unsigned int get_id_in_mesh_0() 
{
	return blockIdx.x;
}

__device__ __forceinline__ unsigned int get_id_in_mesh_1() 
{
	return blockIdx.y;
}

__device__ __forceinline__ unsigned int get_id_in_mesh_2() 
{
	return blockIdx.z;
}

// mesh sizes
__device__ __forceinline__ unsigned int get_mesh_size_0() 
{
	return gridDim.x;
}

__device__ __forceinline__ unsigned int get_mesh_size_1() 
{
	return gridDim.y;
}

__device__ __forceinline__ unsigned int get_mesh_size_2() 
{
	return gridDim.z;
}

// bundle ids
__device__ __forceinline__ unsigned int get_id_in_bundle_0() 
{
	return threadIdx.x;
}

__device__ __forceinline__ unsigned int get_id_in_bundle_1() 
{
	return threadIdx.y;
}

__device__ __forceinline__ unsigned int get_id_in_bundle_2() 
{
	return threadIdx.z;
}

// bundle sizes
__device__ __forceinline__ unsigned int get_bundle_size_0() 
{
	return blockDim.x;
}

__device__ __forceinline__ unsigned int get_bundle_size_1() 
{
	return blockDim.y;
}

__device__ __forceinline__ unsigned int get_bundle_size_2() 
{
	return blockDim.z;
}

#define AURA_KERNEL extern "C" __global__
#define AURA_GLOBAL 
#define AURA_DEVICE_FUNCTION __device__
#define AURA_SHARED __shared__
#define AURA_SYNC __syncthreads();
#define AURA_RESTRICT __restrict__

#endif // AURA_BACKEND_CUDA_KERNEL_HELPER_HPP

