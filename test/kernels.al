#ifdef AURA_BASE_CUDA
__global__ void  add(int *a, int *b, int *c ) 
{
        int tid = blockIdx.x;    
#endif
#ifdef AURA_BASE_OPENCL
__kernel void  add
        (__global int *a, __global int *b, __global int *c)
{
        int tid = get_global_id(0);
#endif
#ifdef AURA_BASE_METAL
#include <metal_stdlib>
using namespace metal;
kernel void  add
        (device int *a [[buffer(0)]], 
        device int *b [[buffer(1)]], 
        device int *c [[buffer(2)]],
        const uint tid [[thread_position_in_grid]])
{
#endif
        c[tid] = a[tid] + b[tid];
}

