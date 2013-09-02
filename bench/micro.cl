__kernel void noarg() {}

__kernel void simple_add(__global float * A) {
  int id = get_global_id(0) * get_global_size(0) + get_local_id(1);
  A[id] += 1.0;
}

