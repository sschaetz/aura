#ifndef AURA_BACKEND_CUDA_KERNEL_ATOMIC_HPP
#define AURA_BACKEND_CUDA_KERNEL_ATOMIC_HPP

__device__ static __forceinline__ float atomic_addf(float* address, float val) {
	return atomicAdd(address, val);
}

__device__ static __forceinline__ float atomic_maxf(float* address, float val) {

	int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;

    do {

        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));

    } while (assumed != old);

    return __int_as_float(old);
}

#endif // AURA_BACKEND_CUDA_KERNEL_ATOMIC_HPP

