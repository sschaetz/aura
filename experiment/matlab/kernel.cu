
__device__ __forceinline__ unsigned int get_mesh_id() 
{
  return (gridDim.y*gridDim.x*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x) *
    (blockDim.z*blockDim.y*blockDim.x) +
    blockDim.y*blockDim.x*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
}

extern "C" __global__ void kern_c2i(int count, float* dst, float* src1, float* src2)
{
	unsigned int i = get_mesh_id(); 
	if (i < count) {
		dst[i*2] = src1[i];
		dst[i*2+1] = src2[i];
	}	
}

extern "C" __global__ void kern_i2c(int count, float* dst1, float* dst2, float* src)
{
	unsigned int i = get_mesh_id(); 
	if (i < count) {
		dst1[i] = src[2*i];
		dst2[i] = src[2*i+1];
	}		
}

