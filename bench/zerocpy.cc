#include <boost/aura/backend.hpp>

AURA_KERNEL void compute(AURA_GLOBAL float* dst,
                         AURA_GLOBAL float* src)
{
	int id = get_mesh_id();
	float tmp = src[id];

#pragma unroll
	for (int i=0; i<8; i++) {
		float x = tmp;
		tmp = tmp * tmp + tmp;
		tmp = tmp / x;
	}
	dst[id] = tmp-8.;
}

