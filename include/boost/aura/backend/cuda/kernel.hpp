#ifndef AURA_BACKEND_CUDA_KERNEL_HPP
#define AURA_BACKEND_CUDA_KERNEL_HPP

#include <cuda.h>
#include <boost/aura/backend/cuda/call.hpp>
#include <boost/aura/backend/cuda/device.hpp>
#include <boost/aura/backend/cuda/module.hpp>

namespace boost
{
namespace aura {
namespace backend_detail {
namespace cuda {

/// kernel handle
typedef CUfunction kernel;

/**
 * @brief create a kernel
 *
 * @param m module that contains the kernel
 * @param kernel_name name of the kernel
 */
inline kernel create_kernel(module m, const char * kernel_name) {
  kernel k;
  AURA_CUDA_SAFE_CALL(cuModuleGetFunction(&k, m, kernel_name));
  return k;
}

/**
 * @brief print the module build log
 *
 * @param m the module that is built
 * @param d the device the module is built for
 */
inline void print_module_build_log(module & m, const device & d) {
  // FIXME
}


} // cuda
} // backend_detail
} // aura
} // boost

#endif // AURA_BACKEND_CUDA_KERNEL_HPP

