#ifndef AURA_BACKEND_OPENCL_KERNEL_HPP
#define AURA_BACKEND_OPENCL_KERNEL_HPP

#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

/// module handle
typedef cl_program module;

/// kernel handle
typedef cl_kernel kernel;

/**
 * @brief build a kernel module from a source string
 *
 * @param source source string
 * @param length length of the source string
 * @param device device the module is built for
 * @param build_options options for the compiler (optional)
 *
 * @return module reference to compiled module
 */
module build_module_from_source(const char * source, 
    std::size_t length, device & d, const char * build_options=NULL) {
  int errorcode = 0;
  module m = clCreateProgramWithSource(d.get_context(), 1, 
    &source, &length, &errorcode);
  AURA_OPENCL_CHECK_ERROR(errorcode);
  AURA_OPENCL_SAFE_CALL(clBuildProgram(m, 1, &d.get_device(), 
    build_options, NULL, NULL));
  return m;
}


/**
 * @brief create a kernel
 *
 * @param m module that contains the kernel
 * @param kernel_name name of the kernel
 */
kernel create_kernel(module m, const char * kernel_name) {
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
void print_module_build_log(module m, device & d) {
  // from http://stackoverflow.com/a/9467325/244786
  // Determine the size of the log
  std::size_t log_size;
  clGetProgramBuildInfo(m, d.get_device(), 
    CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

  // Allocate memory for the log
  char *log = (char *) malloc(log_size);

  // Get the log
  clGetProgramBuildInfo(m, d.get_device(), CL_PROGRAM_BUILD_LOG, 
    log_size, log, NULL);

  // Print the log
  printf("%s\n", log);
  free(log);
}

} // opencl
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_KERNEL_HPP

