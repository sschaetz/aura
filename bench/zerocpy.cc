#include <aura/backend.hpp>

AURA_KERNEL void copy(AURA_GLOBAL float* dst, 
		AURA_GLOBAL float* src) 
{
	int id = get_mesh_id(); 
	dst[id] = src[id]; 
}

