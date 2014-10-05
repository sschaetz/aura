__device__ inline float2 operator+(float2 a, float2 b) 
{ return make_float2( a.x + b.x, a.y + b.y ); }

__device__ __forceinline__ unsigned int get_mesh_id() {
  return (gridDim.y*gridDim.x*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x) *
    (blockDim.z*blockDim.y*blockDim.x) +
    blockDim.y*blockDim.x*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
}

__device__ __forceinline__ unsigned int get_mesh_size() {
  return gridDim.z*gridDim.y*gridDim.x * blockDim.z*blockDim.y*blockDim.x;
}

__device__ __forceinline__ unsigned int get_bundle_id() {
  return blockDim.y*blockDim.x*threadIdx.z + 
    blockDim.x*threadIdx.y + threadIdx.x;
}
extern "C" __global__ void p2p_4(float2 * dst, float2 * src1,
    float2 * src2, float2 * src3, float2 * src4) {
  int id = get_mesh_id();
  dst[id] = src1[id] + src2[id] + src3[id] + src4[id];
}

extern "C" __global__ void p2p_4_center(float2 * dst, float2 * src1,
    float2 * src2, float2 * src3, float2 * src4,
    unsigned int dim,
    unsigned int offset) {
  int id = offset + blockIdx.x * dim + threadIdx.x;
  dst[id] = src1[id] + src2[id] + src3[id] + src4[id];
}

extern "C" __global__ void p2p_2(float2 * dst, float2 * src1, 
    float2 * src2) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  dst[id] = src1[id] + src2[id];
}

