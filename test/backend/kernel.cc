#include <boost/aura/backend.hpp>

AURA_KERNEL void noarg() {}

AURA_KERNEL void simple_add(AURA_GLOBAL float * A) {
	int id = get_mesh_id(); 
	A[id] += 1.0; 
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

