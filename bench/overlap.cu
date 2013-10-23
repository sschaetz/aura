__device__ inline float2 operator+(float2 a, float2 b) 
{ return make_float2( a.x + b.x, a.y + b.y ); }

extern "C" __global__ void p2p(float2 * dst, float2 * src1,
    float2 * src2, float2 * src3) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  dst[id] = src1[id] + src2[id] + src3[id];
}
