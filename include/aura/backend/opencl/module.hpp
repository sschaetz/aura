#ifndef AURA_BACKEND_OPENCL_MODULE_HPP
#define AURA_BACKEND_OPENCL_MODULE_HPP

#include <fstream>
#include <string>
#include <aura/backend/shared/call.hpp>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

/// module handle
typedef cl_program module;

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

  int errorcode = 0;
  const char * cdata = data.c_str();
  const std::size_t size = data.size();
  
  module m = clCreateProgramWithSource(d.get_backend_context(), 1, 
    &cdata, &size, &errorcode);
  AURA_OPENCL_CHECK_ERROR(errorcode);
  AURA_OPENCL_SAFE_CALL(clBuildProgram(m, 1, &d.get_backend_device(), 
    build_options, NULL, NULL));
  return m;
}

} // opencl
} // backend_detail
} // aura


#endif // AURA_BACKEND_OPENCL_MODULE_HPP

