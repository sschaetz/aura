#ifndef AURA_BACKEND_OPENCL_CALL_HPP
#define AURA_BACKEND_OPENCL_CALL_HPP

#define AURA_OPENCL_SAFE_CALL(call) { \
  int err = call; \
  if (err != CL_SUCCESS) { \
    printf("OpenCL error %d\n", err); \
  } \
} \
/**/

#define AURA_OPENCL_CHECK_ERROR(err) { \
  if (err != CL_SUCCESS) { \
    printf("OpenCL error %d\n", err); \
  } \
} \
/**/

#endif // AURA_BACKEND_OPENCL_CALL_HPP

