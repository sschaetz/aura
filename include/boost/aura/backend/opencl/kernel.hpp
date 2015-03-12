#ifndef AURA_BACKEND_OPENCL_KERNEL_HPP
#define AURA_BACKEND_OPENCL_KERNEL_HPP
#if 0
#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif
#include <boost/aura/backend/opencl/call.hpp>
#include <boost/aura/backend/opencl/device.hpp>


namespace boost
{
namespace aura {
namespace backend_detail {
namespace opencl {

class module;

/// kernel handle
typedef cl_kernel kernel;

/**
 * @brief create a kernel
 *
 * @param m module that contains the kernel
 * @param kernel_name name of the kernel
 */
inline kernel create_kernel(module m, const char * kernel_name) {
  int errorcode = 0;
  kernel k = clCreateKernel(m, kernel_name, &errorcode);
  AURA_OPENCL_CHECK_ERROR(errorcode);
  return k;
}



} // opencl
} // backend_detail
} // aura
} // boost
#endif // 0
#endif // AURA_BACKEND_OPENCL_KERNEL_HPP

