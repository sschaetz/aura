#ifndef AURA_BACKEND_OPENCL_MODULE_HPP
#define AURA_BACKEND_OPENCL_MODULE_HPP

#include <fstream>
#include <string>
#include <cstring>
#include <boost/move/move.hpp>

namespace boost {
namespace aura {
namespace backend_detail {
namespace opencl {

class module;
class device;
const cl_context & get_backend_context(const device & d);
const cl_device_id & get_backend_device(const device & d);
typedef cl_kernel kernel;

module create_module_from_file(const char * filename, device & d,
		const char * build_options);

void print_module_build_log(const module & m, const device & d);


class module
{

private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(module)


public:
	inline explicit module() : device_(nullptr), program_(nullptr) {}

	inline explicit module(const char * str, device & d,
			 const char * build_options=NULL)
			: device_(&d)
	{
		int errorcode = 0;
		std::size_t len = strlen(str);
		program_ = clCreateProgramWithSource(get_backend_context(*device_), 1,
						     &str, &len, &errorcode);
		AURA_OPENCL_CHECK_ERROR(errorcode);
		AURA_OPENCL_SAFE_CALL(clBuildProgram(program_, 1, &(get_backend_device(*device_)),
						     build_options, NULL, NULL));
		print_module_build_log(*this, *device_);
	}

	/**
	 * destoy module
	 */
	inline ~module()
	{
		finalize();
	}

	/**
	* move constructor, move module here, invalidate other
	*
	* @param m module to move here
	*/
	module(BOOST_RV_REF(module) m) :
		device_(m.device_),
		program_(m.program_),
		kernels_(std::move(m.kernels_))
	{
		m.device_ = nullptr;
		m.program_ = nullptr;
		m.kernels_.clear();
	}

	/**
	* move assignment, move device information here, invalidate other
	*
	* @param d device to move here
	*/
	module& operator=(BOOST_RV_REF(module) m)
	{
		finalize();
		device_ = m.device_;
		program_ = m.program_;
		kernels_ = std::move(m.kernels_);

		m.device_ = nullptr;
		m.program_ = nullptr;
		m.kernels_.clear();
		return *this;
	}


	inline kernel & get_kernel(const char * kernel_name)
	{
		auto it = kernels_.find(kernel_name);
		if (kernels_.end() == it) {
			int errorcode = 0;
			kernel k = clCreateKernel(program_, kernel_name, &errorcode);
			AURA_OPENCL_CHECK_ERROR(errorcode);
			auto it2 = kernels_.insert(
				std::make_pair(kernel_name, k));
			it = it2.first;
		}
		return it->second;
	}




	// access the device
	const device& get_device() const
	{
		return *device_;
	}
	device& get_device()
	{
		return *device_;
	}

	// access the program
	const cl_program get_backend_module() const
	{
		return program_;
	}
	cl_program get_backend_module()
	{
		return program_;
	}


private:
	/// finalize object (called from dtor and move assign)
	void finalize()
	{
		for (auto& it : kernels_) {
			clReleaseKernel(it.second);
		}
		if (nullptr != program_) {
			clReleaseProgram(program_);
		}
	}

private:
	device * device_;
	cl_program program_;
	std::unordered_map<std::string, kernel> kernels_;
};

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
	return std::move(module(data.c_str(), d, build_options));
}


/**
 * @brief print the module build log
 *
 * @param m the module that is built
 * @param d the device the module is built for
 */

inline void print_module_build_log(const module & m, const device & d) {
  // from http://stackoverflow.com/a/9467325/244786
  // Determine the size of the log
  std::size_t log_size;
  clGetProgramBuildInfo(m.get_backend_module(), get_backend_device(d),
    CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

  // Allocate memory for the log
  char *log = (char *) malloc(log_size);

  // Get the log
  clGetProgramBuildInfo(m.get_backend_module(), get_backend_device(d), CL_PROGRAM_BUILD_LOG,
    log_size, log, NULL);

  // Print the log
  if (strncmp("\n\0", log, 2) != 0) {
	printf("%s\n", log);
  }
  free(log);
}


} // opencl
} // backend_detail
} // aura
} // boost


#endif // AURA_BACKEND_OPENCL_MODULE_HPP

