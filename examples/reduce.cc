#include <boost/aura/backend.hpp>

AURA_KERNEL void red1(AURA_GLOBAL float* A, 
		AURA_GLOBAL float* r) {
	unsigned int id = get_mesh_id();
	unsigned int bid = get_bundle_id();	
	AURA_SHARED float sm[16];
	sm[bid] = A[id];
	AURA_SYNC;	
	
	if (bid < 8) 
		sm[bid] += sm[bid+8];
	AURA_SYNC;

	if (bid < 4) 
		sm[bid] += sm[bid+4];
	AURA_SYNC;

	if (bid < 2) 
		sm[bid] += sm[bid+2];
	AURA_SYNC;

	if (bid < 1) {
		sm[0] += sm[1];
		atomic_addf(r, sm[0]);
	}
}

