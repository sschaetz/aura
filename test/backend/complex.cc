#include <boost/aura/backend.hpp>

AURA_KERNEL void complex_single(AURA_GLOBAL float * A) 
{
	int id = get_mesh_id(); 
	cfloat c = make_cfloat((float)id+1, (float)(id+1)*0.1);
	cfloat c1 = caddf(conjf(make_cfloat(cimagf(c), crealf(c))), c);
	cfloat c2 = csubf(conjf(make_cfloat(cimagf(c), crealf(c))), c);
	cfloat c3 = cdivf(cmulf(c1, c2), c);
	A[id] = cabsf(c3);
}

