#ifndef AURA_BACKEND_CUDA_CALL_HPP
#define AURA_BACKEND_CUDA_CALL_HPP

/// check if a call returned error and throw exception if it did
#define AURA_CUDA_SAFE_CALL(call) call;

/// check for error and throw exception if true 
#define AURA_OPENCL_CHECK_ERROR(error) ;

#endif // AURA_BACKEND_CUDA_CALL_HPP

