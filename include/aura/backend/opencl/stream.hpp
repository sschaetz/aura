#ifndef AURA_BACKEND_OPENCL_STREAM_HPP
#define AURA_BACKEND_OPENCL_STREAM_HPP


#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>

namespace aura {
namespace backend_detail {
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
  stream s = clCreateCommandQueue(c, d, 0, &errorcode);
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
 * wait (block) until all operations in the stream are finished
 *
 * @param s stream
 */
inline void stream_synchronize(stream s) {
  cl_event event;
  AURA_OPENCL_SAFE_CALL(clEnqueueMarker(s, &event));
  AURA_OPENCL_SAFE_CALL(clEnqueueWaitForEvents(s, 1, &event));
}

/**
 * destroy stream
 *
 * @param stream stream that should be destroyed
 */
inline void stream_destroy(stream s) {
  AURA_OPENCL_SAFE_CALL(clReleaseCommandQueue(s));
}

} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_STREAM_HPP

