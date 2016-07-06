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


// Alang version of add kernel.

AURA_KERNEL void add_alang(
        AURA_DEVMEM float* a, 
        AURA_DEVMEM float* b, 
        AURA_DEVMEM float* c
        AURA_MESH_ID_ARG)
{
        c[AURA_MESH_ID_0] = a[AURA_MESH_ID_0] + b[AURA_MESH_ID_0];
}

