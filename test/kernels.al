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
        c[tid] = a[tid] + b[tid];
}

