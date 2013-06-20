#ifndef AURA_BACKEND_CUDA_CALL_HPP
#define AURA_BACKEND_CUDA_CALL_HPP

/// check if a call returned error and throw exception if it did
#define AURA_CUDA_SAFE_CALL(call) { \
  CUresult err = call; \
  if (err != CUDA_SUCCESS) { \
    printf("CUDA error %d\n", err); \
  } \
  else { printf(" no error! %s %s:%d\n ", #call, __FILE__, __LINE__); } \
} \
/**/

/// check for error and throw exception if true 
#define AURA_CUDA_CHECK_ERROR(error) { \
  if (err != CUDA_SUCCESS) { \
    printf("CUDA error %d\n", err); \
  } \
} \
/**/

#endif // AURA_BACKEND_CUDA_CALL_HPP

