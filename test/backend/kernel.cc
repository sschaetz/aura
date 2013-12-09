#include <aura/backend.hpp>

AURA_KERNEL void noarg() {}

AURA_KERNEL void simple_add(AURA_GLOBAL float * A) {
  int id = get_mesh_id(); 
  A[id] += 1.0; 
}

