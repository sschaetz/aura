#ifndef AURA_BACKEND_OPENCL_STREAM_HPP
#define AURA_BACKEND_OPENCL_STREAM_HPP


#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>

namespace aura {
namespace backend {
namespace opencl {

/// device handle
typedef cl_command_queue stream;

/**
 * create stream for device context
 *
 * @param device device to create stream for
 * @param context context to create stream for (must match device)
 * @return the stream handle
 */
inline stream stream_create(device d, context c) {
  int errorcode = 0;
  stream s = clCreateCommandQueue(c, d, NULL, &errorcode);
  AURA_OPENCL_CHECK_ERROR(errorcode);
  return s;
}

/**
 * create default stream for device context
 *
 * @param device device to create stream for
 * @param context context to create stream for (must match device)
 * @return the default stream handle
 */
inline stream stream_create_default(device d, context c) {
  return stream_create(d, c);
}

/**
 * destroy stream
 *
 * @param stream stream that should be destroyed
 */
inline void stream_destroy(stream s) {
  AURA_CUDA_SAFE_CALL(clReleaseCommandQueue(s));
}

} // opencl 
} // backend
} // aura

#endif // AURA_BACKEND_OPENCL_STREAM_HPP

