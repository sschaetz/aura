#ifndef AURA_BACKEND_OPENCL_KERNEL_HPP
#define AURA_BACKEND_OPENCL_KERNEL_HPP

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif
#include <boost/aura/backend/opencl/call.hpp>
#include <boost/aura/backend/opencl/device.hpp>
#include <boost/aura/backend/opencl/module.hpp>

namespace boost
{
namespace aura {
namespace backend_detail {
namespace opencl {

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

/**
 * @brief print the module build log
 *
 * @param m the module that is built
 * @param d the device the module is built for
 */
inline void print_module_build_log(module & m, const device & d) {
  // from http://stackoverflow.com/a/9467325/244786
  // Determine the size of the log
  std::size_t log_size;
  clGetProgramBuildInfo(m, d.get_backend_device(), 
    CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

  // Allocate memory for the log
  char *log = (char *) malloc(log_size);

  // Get the log
  clGetProgramBuildInfo(m, d.get_backend_device(), CL_PROGRAM_BUILD_LOG, 
    log_size, log, NULL);

  // Print the log
  printf("%s\n", log);
  free(log);
}

} // opencl
} // backend_detail
} // aura
} // boost

#endif // AURA_BACKEND_OPENCL_KERNEL_HPP

