#ifndef AURA_BACKEND_CUDA_KERNEL_HPP
#define AURA_BACKEND_CUDA_KERNEL_HPP

#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>
#include <aura/backend/cuda/module.hpp>

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
kernel create_kernel(module m, const char * kernel_name) {
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
void print_module_build_log(module & m, device & d) {
  // FIXME
}


} // cuda
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_KERNEL_HPP

