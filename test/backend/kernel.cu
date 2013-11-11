#include <aura/backend.hpp>

extern "C" __global__ void noarg() {}

extern "C" __global__ void simple_add(float * A) {
  int id = get_mesh_id();
  A[id] += 1.0;
}


