#ifndef AURA_BACKEND_CUDA_MODULE_HPP
#define AURA_BACKEND_CUDA_MODULE_HPP

#include <fstream>
#include <string>
#include <cuda.h>
#include <aura/backend/shared/call.hpp>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

/// module handle
typedef CUmodule module;

/**
 * @brief build a kernel module from a source file 
 *
 * @param filename name of .cl or .ptx of .fatbin or .cubin 
 * @param device device the module is built for
 * @param build_options options for the compiler (optional)
 *
 * @return module reference to compiled module
 */
module create_module_from_file(const char * filename, device & d, 
  const char * build_options=NULL) {
  std::ifstream in(filename, std::ios::in);
  AURA_CHECK_ERROR(in);
  in.seekg(0, std::ios::end);
  std::string data;
  data.resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&data[0], data.size());
  in.close();
  
  // build for device by setting context and JIT argument
  d.set();
 
  const std::size_t num_options = 1;  
  CUjit_option options[num_options];
  void * values[num_options];

  // set jit target from context (which we got by setting the device)
  options[0] = CU_JIT_TARGET_FROM_CUCONTEXT;
  values[0] = NULL;
  
  module m;
  AURA_CUDA_SAFE_CALL(cuModuleLoadDataEx(&m, data.c_str(), 
    num_options, options, values));
  d.unset();

  return m;
}

} // cuda
} // backend_detail
} // aura


#endif // AURA_BACKEND_CUDA_MODULE_HPP
