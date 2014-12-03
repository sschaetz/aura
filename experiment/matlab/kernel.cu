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
		dst1[i] = src[2*i];
		dst2[i] = src[2*i+1];
	}		
}

inline __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

AURA_KERNEL void kern_axpy(AURA_GLOBAL int count, float2 a,
		AURA_GLOBAL float2* X, AURA_GLOBAL float2* Y)
{
	unsigned int i = get_mesh_id(); 
	if (i < count) {
		Y[i] = a*X[i] + Y[i];
	}		
}

