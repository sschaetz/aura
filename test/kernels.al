#ifdef AURA_BASE_CUDA
__global__ void  add(int *a, int *b, int *c ) 
{
        int tid = blockIdx.x;    
        c[tid] = a[tid] + b[tid];
}
#endif

