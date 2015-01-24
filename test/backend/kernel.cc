#include <boost/aura/backend.hpp>

AURA_KERNEL void donothing(AURA_GLOBAL float * A) {
	int id = get_mesh_id(); 
	(void)id;
}

AURA_KERNEL void simple_add(AURA_GLOBAL float * A) {
	int id = get_mesh_id(); 
	A[id] += 1.0f; 
}

AURA_KERNEL void simple_shared(AURA_GLOBAL float * A) {
  
	int id = get_mesh_id();
	int bid = get_bundle_id();
	
	AURA_SHARED float sm[8];
	
	sm[bid] = A[id]+(float)get_mesh_id();
	AURA_SYNC;
	
	// write back in reverse
	bid = get_bundle_size() - bid - 1;
	A[id] = sm[bid]; 
}

AURA_KERNEL void simple_atomic(AURA_GLOBAL float * A) {
	atomic_addf(A, 1.0f);  
}

