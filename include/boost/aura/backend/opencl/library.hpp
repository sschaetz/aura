#ifndef AURA_BACKEND_OPENCL_LIBRARY_HPP
#define AURA_BACKEND_OPENCL_LIBRARY_HPP

#include <boost/aura/backend/opencl/device.hpp>

#include <fstream>
#include <string>
#include <cstring>
#include <boost/move/move.hpp>

namespace boost {
namespace aura {
namespace backend_detail {
namespace opencl {

typedef cl_kernel kernel;

class library
{

private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(library)


public:
    /// create empty library (default ctor)
    inline explicit library() : device_(nullptr), program_(nullptr) {}

    /// create library from a string
    /// @param str string to compile library from
    /// @param d device to compile library for
    /// @param buil_options build options to consider for compilation
	inline explicit library(const char * str, device & d,
			 const char * build_options=NULL)
			: device_(&d)
	{
        compile(str, build_options);
	}

    /// create library from a stream
    /// @param stream stream containing library source code to be compiled
    /// @param d device to compile library for
    /// @param buil_options build options to consider for compilation
    template <typename STREAM>
    inline explicit library(STREAM& stream, device&d, const char* build_options=NULL)
            : device_(&d)
    {
        AURA_CHECK_ERROR(stream);
        stream.seekg(0, std::ios::end);
        std::string str;
        str.resize(stream.tellg());
        stream.seekg(0, std::ios::beg);
        stream.read(&str[0], str.size());
        stream.close();
        compile(str.c_str(), build_options);
    }

	/**
	 * destoy library
	 */
	inline ~library()
	{
		finalize();
	}

	/**
	 * move constructor, move library here, invalidate other
	 *
	 * @param m library to move here
	 */
	library(BOOST_RV_REF(library) m) :
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
	library& operator=(BOOST_RV_REF(library) m)
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

    /// Create a kernel from the library
    /// @param kernel_name name of kernel to be created
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
	const cl_program get_backend_library() const
	{
		return program_;
	}

	cl_program get_backend_library()
	{
		return program_;
	}

    /**
     * @brief print the library build log
     */
    inline void print_library_build_log()
    {
      // from http://stackoverflow.com/a/9467325/244786
      // Determine the size of the log
      std::size_t log_size;
      clGetProgramBuildInfo(get_backend_library(), device_->get_backend_device(),
        CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

      // Allocate memory for the log
      char *log = (char *) malloc(log_size);

      // Get the log
      clGetProgramBuildInfo(program_,
              device_->get_backend_device(), CL_PROGRAM_BUILD_LOG,
              log_size, log, NULL);

      // Print the log
      if (strncmp("\n\0", log, 2) != 0) {
        printf("%s\n", log);
      }
      free(log);
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

    /// compile string to library
    /// @param str string to be compiled
    /// @param build_options build options
    void compile(const char * str, const char * build_options=NULL)
	{
		int errorcode = 0;
		std::size_t len = strlen(str);
		program_ = clCreateProgramWithSource(
                device_->get_backend_context(), 1,
				&str, &len, &errorcode);
		AURA_OPENCL_CHECK_ERROR(errorcode);
		AURA_OPENCL_SAFE_CALL(clBuildProgram(program_, 1,
                    &device_->get_backend_device(),
					build_options, NULL, NULL));
		print_library_build_log();
	}

	device * device_;
	cl_program program_;
	std::unordered_map<std::string, kernel> kernels_;
};

/// compile file to library
/// @param kernel_file path to kernel file to be compiled
/// @param d device kernel file is compiled for
/// @param build_options build options
library make_library_from_file(const char* kernel_file,
        device& d, const char* build_options = NULL)
{
    std::ifstream s;
    s.open(kernel_file);
    return library(s, d, build_options);
}

} // opencl
} // backend_detail
} // aura
} // boost


#endif // AURA_BACKEND_OPENCL_LIBRARY_HPP

