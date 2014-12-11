#ifndef AURA_BACKEND_OPENCL_KERNEL_HELPER_HPP
#define AURA_BACKEND_OPENCL_KERNEL_HELPER_HPP

inline unsigned int get_mesh_id() {
	return get_global_size(1)*get_global_size(0)*get_global_id(2) +
		get_global_size(0)*get_global_id(1) + get_global_id(0);
}

inline unsigned int get_mesh_size() {
	return get_global_size(0)*get_global_size(1)*get_global_size(2);
}

inline unsigned int get_bundle_id() {
	return get_local_size(1)*get_local_size(0)*get_local_id(2) + 
		get_local_size(0)*get_local_id(1) + get_local_id(0);
}

inline unsigned int get_bundle_size() {
	return get_local_size(0)*get_local_size(1)*get_local_size(2);
}

#define AURA_KERNEL __kernel
#define AURA_GLOBAL __global
#define AURA_DEVICE_FUNCTION
#define AURA_SHARED __local 
#define AURA_SYNC barrier(CLK_LOCAL_MEM_FENCE);

#endif // AURA_BACKEND_OPENCL_KERNEL_HELPER_HPP

