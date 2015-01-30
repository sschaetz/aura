#ifndef AURA_BACKEND_OPENCL_CALL_HPP
#define AURA_BACKEND_OPENCL_CALL_HPP

#include <stdio.h>
#include <sstream>

#define AURA_OPENCL_SAFE_CALL(call) { \
  int err = call; \
  if (err != CL_SUCCESS) { \
    printf("OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__ ); \
  } \
} \
/**/

#define AURA_CLFFT_SAFE_CALL(call) { \
	int err = call; \
	if (err != CLFFT_SUCCESS) { \
		std::ostringstream os; \
		os << "clFFT error " << err << " file " << \
			__FILE__ << " line " << __LINE__; \
		throw os.str(); \
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

