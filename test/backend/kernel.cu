__global__ void noarg() {}

__global__ void simple_add(float * A) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  A[id += 1.0;
}


