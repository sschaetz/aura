#include <boost/aura/backend.hpp>


AURA_KERNEL void kern_c2i(AURA_GLOBAL int count, AURA_GLOBAL float* src1, 
		AURA_GLOBAL float* src2, AURA_GLOBAL float* dst)
{
	unsigned int i = get_mesh_id();
	if (i < count) {
		dst[i*2] = src1[i];
		dst[i*2+1] = src2[i];
	}	
}

AURA_KERNEL void kern_i2c(AURA_GLOBAL int count, AURA_GLOBAL float* src,
		AURA_GLOBAL float* dst1, AURA_GLOBAL float* dst2)
{
	unsigned int i = get_mesh_id(); 
	if (i < count) {
		dst1[i] = src[2*i]*3;
		dst2[i] = src[2*i+1]*2;
	}		
}



