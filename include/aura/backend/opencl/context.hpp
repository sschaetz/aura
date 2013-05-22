#ifndef AURA_BACKEND_OPENCL_CONTEXT_HPP
#define AURA_BACKEND_OPENCL_CONTEXT_HPP

#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>

namespace aura {
namespace backend {
namespace opencl {

/// context handle
typedef cl_context context;


/**
 * create context for device 
 *
 * @param d the device handle 
 * @return the context handle
 */
inline context create_context(device d) {
  int errorcode = 0;
  context c = clCreateContext(NULL, 1, &d, NULL, NULL, &errorcode);
  AURA_OPENCL_CHECK_ERROR(errorcode);
  return c;
}

} // opencl 
} // backend
} // aura

#endif // AURA_BACKEND_OPENCL_CONTEXT_HPP

