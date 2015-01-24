#ifndef AURA_BACKEND_OPENCL_KERNEL_HELPER_HPP
#define AURA_BACKEND_OPENCL_KERNEL_HELPER_HPP

inline unsigned int get_mesh_id() 
{
	return get_global_size(1)*get_global_size(0)*get_global_id(2) +
		get_global_size(0)*get_global_id(1) + get_global_id(0);
}

inline unsigned int get_id_in_mesh() 
{
	return get_mesh_id();
}

inline unsigned int get_mesh_size() 
{
	return get_global_size(0)*get_global_size(1)*get_global_size(2);
}

inline unsigned int get_bundle_id() 
{
	return get_local_size(1)*get_local_size(0)*get_local_id(2) + 
		get_local_size(0)*get_local_id(1) + get_local_id(0);
}

inline unsigned int get_id_in_bundle() 
{
	return get_bundle_id();
}

inline unsigned int get_bundle_size() 
{
	return get_local_size(0)*get_local_size(1)*get_local_size(2);
}


// mesh ids
inline unsigned int get_id_in_mesh_0() 
{
	return get_group_id(0);
}

inline unsigned int get_id_in_mesh_1() 
{
	return get_group_id(1);
}

inline unsigned int get_id_in_mesh_2() 
{
	return get_group_id(2);
}

// mesh sizes
inline unsigned int get_mesh_size_0() 
{
	return get_global_size(0);
}

inline unsigned int get_mesh_size_1() 
{
	return get_global_size(1);
}

inline unsigned int get_mesh_size_2() 
{
	return get_global_size(2);
}


// bundle ids
inline unsigned int get_id_in_bundle_0() 
{
	return get_local_id(0);
}

inline unsigned int get_id_in_bundle_1() 
{
	return get_local_id(1);
}

inline unsigned int get_id_in_bundle_2() 
{
	return get_local_id(2);
}

// bundle sizes
inline unsigned int get_bundle_size_0() 
{
	return get_local_size(0);
}

inline unsigned int get_bundle_size_1() {
	return get_local_size(1);
}

inline unsigned int get_bundle_size_2() {
	return get_local_size(2);
}



#define AURA_KERNEL __kernel
#define AURA_GLOBAL __global
#define AURA_DEVICE_FUNCTION
#define AURA_SHARED __local 
#define AURA_SYNC barrier(CLK_LOCAL_MEM_FENCE);
#define AURA_RESTRICT restrict

#endif // AURA_BACKEND_OPENCL_KERNEL_HELPER_HPP

