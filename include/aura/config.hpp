
#ifndef AURA_CONFIG_HPP
#define AURA_CONFIG_HPP

/// maximum size of elements in svec specialized types
#define AURA_SVEC_MAX_SIZE 16

/// maximum dimensionality of a MESH 
#define AURA_MAX_MESH_DIMS 4

/// maximum dimensionality of a BUNDLE 
#define AURA_MAX_BUNDLE_DIMS 4

/// maximum number of possible kernel arguments
#define AURA_MAX_KERNEL_ARGS 12

/// default threads if user does not want to specify mesh and bundle
#define AURA_CUDA_NUM_DEFAULT_THREADS 512

/// if kernel size is calculated, these are the max allowed values 
#define AURA_CUDA_MAX_BUNDLE 512
#define AURA_CUDA_MAX_MESH0 65536 
#define AURA_CUDA_MAX_MESH1 65536 
#define AURA_CUDA_MAX_MESH2 65536 

#define AURA_OPENCL_MAX_BUNDLE 256 
#define AURA_OPENCL_MAX_MESH0 1024 
#define AURA_OPENCL_MAX_MESH1 1024 
#define AURA_OPENCL_MAX_MESH2 1024 

#endif // AURA_CONFIG_HPP 

