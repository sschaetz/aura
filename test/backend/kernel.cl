#include <aura/backend.hpp>

__kernel void noarg() {}

__kernel void simple_add(__global float * A) {
  int id = get_mesh_id(); 
  A[id] += 1.0; 
}

