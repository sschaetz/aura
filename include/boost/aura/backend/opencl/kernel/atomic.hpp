#ifndef AURA_BACKEND_OPENCL_KERNEL_ATOMIC_HPP
#define AURA_BACKEND_OPENCL_KERNEL_ATOMIC_HPP

// adapted from http://suhorukov.blogspot.de/2011/12/opencl-11-atomic-operations-on-floating.html
// Copyright Igor Suhorukov
inline float atomic_addf(volatile __global float* address, float val) {
	union uif 
	{
		unsigned int ui;
		float f;
	};

	union uif newvalue;
	union uif oldvalue;
	do {
		oldvalue.f = *address;
		newvalue.f = oldvalue.f + val;
	} while (atomic_cmpxchg((volatile __global unsigned int *)address, 
				oldvalue.ui, newvalue.ui) != oldvalue.ui);
	return oldvalue.f;
}

#endif // AURA_BACKEND_OPENCL_KERNEL_ATOMIC_HPP

