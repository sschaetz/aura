#ifndef AURA_BACKEND_CUDA_KERNEL_ATOMIC_HPP
#define AURA_BACKEND_CUDA_KERNEL_ATOMIC_HPP

__device__ static __forceinline__ float atomic_addf(float* address, float val) {
	return atomicAdd(address, val);
}

#endif // AURA_BACKEND_CUDA_KERNEL_ATOMIC_HPP

