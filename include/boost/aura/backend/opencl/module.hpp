#ifndef AURA_BACKEND_OPENCL_MODULE_HPP
#define AURA_BACKEND_OPENCL_MODULE_HPP

#include <fstream>
#include <string>
#include <cstring>
#include <boost/aura/backend/shared/call.hpp>
#include <boost/aura/backend/opencl/call.hpp>
#include <boost/aura/backend/opencl/device.hpp>

namespace boost
{
namespace aura {
namespace backend_detail {
namespace opencl {

/// module handle
typedef cl_program module;

/**
 * @brief build a kernel module from a source string 
 *
 * @ param str string containing kernel source code
 * @param device device the module is built for
 * @param build_options options for the compiler (optional)
 *
 * @return module reference to compiled module
 */
inline module create_module_from_string(const char * str, device & d,
		const char * build_options=NULL) {
	int errorcode = 0;
	std::size_t len = strlen(str);
	module m = clCreateProgramWithSource(d.get_backend_context(), 1,
			&str, &len, &errorcode);
	AURA_OPENCL_CHECK_ERROR(errorcode);
	AURA_OPENCL_SAFE_CALL(clBuildProgram(m, 1, &d.get_backend_device(),
				build_options, NULL, NULL));
	return m;
}


/**
 * @brief build a kernel module from a source file 
 *
 * @param filename name of .cl or .ptx of .fatbin or .cubin 
 * @param device device the module is built for
 * @param build_options options for the compiler (optional)
 *
 * @return module reference to compiled module
 */
inline module create_module_from_file(const char * filename, device & d,
		const char * build_options=NULL) {
	std::ifstream in(filename, std::ios::in);
	AURA_CHECK_ERROR(in);
	in.seekg(0, std::ios::end);
	std::string data;
	data.resize(in.tellg());
	in.seekg(0, std::ios::beg);
	in.read(&data[0], data.size());
	in.close();
	return create_module_from_string(data.c_str(), d, build_options);
}

} // opencl
} // backend_detail
} // aura
} // boost


#endif // AURA_BACKEND_OPENCL_MODULE_HPP

