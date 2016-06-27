#ifdef AURA_BASE_CUDA
extern "C" __global__ void add(float *a, float *b, float *c ) 
{
        unsigned int tid = blockIdx.x;    
#endif
#ifdef AURA_BASE_OPENCL
__kernel void add
        (__global float *a, __global float *b, __global float *c)
{
        uint tid = get_global_id(0);
#endif
#ifdef AURA_BASE_METAL
#include <metal_stdlib>
using namespace metal;
kernel void add
        (const device float* a [[ buffer(0) ]], 
        const device float* b [[ buffer(1) ]], 
        device float* c [[ buffer(2) ]],
        const uint tid [[ thread_position_in_grid ]])
{
#endif
        c[tid] = a[tid] + b[tid];
}

