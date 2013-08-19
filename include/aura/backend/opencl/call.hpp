#ifndef AURA_BACKEND_OPENCL_CALL_HPP
#define AURA_BACKEND_OPENCL_CALL_HPP

#define AURA_OPENCL_SAFE_CALL(call) { \
  int err = call; \
  if (err != CL_SUCCESS) { \
    printf("OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__ ); \
  } \
} \
/**/

#define AURA_OPENCL_CHECK_ERROR(err) { \
  if (err != CL_SUCCESS) { \
    printf("OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__ ); \
  } \
} \
/**/

#endif // AURA_BACKEND_OPENCL_CALL_HPP

