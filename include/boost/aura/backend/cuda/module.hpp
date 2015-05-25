#ifndef AURA_BACKEND_CUDA_MODULE_HPP
#define AURA_BACKEND_CUDA_MODULE_HPP

#include <fstream>
#include <string>
#include <cuda.h>
#include <nvrtc.h>
#include <boost/aura/backend/shared/call.hpp>
#include <boost/aura/backend/cuda/call.hpp>
#include <boost/aura/backend/cuda/context.hpp>

namespace boost {
namespace aura {
namespace backend_detail {
namespace cuda {


class module;
class device;

typedef CUfunction kernel;

module create_module_from_file(const char * filename, device & d,
		const char * build_options);

void print_module_build_log(const module & m, const device & d);

void set(device& d);
void unset(device& d);

inline context* get_contex();

class module
{

private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(module)


public:
	inline explicit module()
		: context_(nullptr)
		, program_(nullptr)
		, log_(nullptr)
	{}

	inline explicit module(const char * str, device & d,
			 const char * build_options=NULL)
		: context_(get_context(d)
		, log_(nullptr)
	{
		set(d);

		// Create and compile.
		nvrtcProgram program;
		nvrtcCreateProgram(program, str, NULL, 0, NULL, NULL);
		nvrtcCompileProgram(program, 0, NULL);

		// Optain PTX.
		size_t ptx_size;
		nvrtcGetPTXSize(program, &ptx_size);
		char* ptx = new char[ptx_size];
		nvrtcGetPTX(program, ptx);

		// Store the log.
		size_t logSize;
		nvrtcGetProgramLogSize(program, &logSize);
		log_ = new char[logSize];
		nvrtcGetProgramLog(program, log_);

		// Program not needed any more.
		nvrtcDestroyProgram(&program);

		// build for device by setting context and JIT argument
		const std::size_t num_options = 1;
		CUjit_option options[num_options];
		void * values[num_options];

		// set jit target from context (which we got by setting the device)
		options[0] = CU_JIT_TARGET_FROM_CUCONTEXT;
		values[0] = NULL;

		AURA_CUDA_SAFE_CALL(cuModuleLoadDataEx(&program_, ptx, num_options,
					options, values));
		unset(d);
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
	module(BOOST_RV_REF(module) m)
		: context_(m.context_)
		, program_(m.program_)
		, kernels_(std::move(m.kernels_))
		, log_(m.log_)
	{
		m.context_= nullptr;
		m.program_ = nullptr;
		m.kernels_.clear();
		m.log_ = nullptr;
	}

	/**
	* move assignment, move module information here, invalidate other
	*
	* @param m module to move here
	*/
	module& operator=(BOOST_RV_REF(module) m)
	{
		finalize();
		context_ = m.context_;
		program_ = m.program_;
		kernels_ = std::move(m.kernels_);
		log_ = m.log_;
		m.finalize();
		return *this;
	}


	inline kernel& get_kernel(const char* kernel_name)
	{
		auto it = kernels_.find(kernel_name);
		if (kernels_.end() == it)
		{
			kernel k;
			AURA_CUDA_SAFE_CALL(cuModuleGetFunction(&k, program_,
						kernel_name));
			auto it2 = kernels_.insert(
				std::make_pair(kernel_name, k));
			it = it2.first;
		}
		return it->second;
	}




	// access the device
	const detail::context& get_context() const
	{
		return *context_;
	}
	detail::context& get_context()
	{
		return *context_;
	}

	// access the program
	const CUmodule get_backend_module() const
	{
		return program_;
	}
	CUmodule get_backend_module()
	{
		return program_;
	}

	char* get_build_log() const
	{
		return log_;
	}

private:
	/// finalize object (called from dtor and move assign)
	void finalize()
	{
		if (nullptr != program_)
		{
			context_->set();
			cuModuleUnload(program_);
			context_->unset();
		}
		if (nullptr != log_)
		{
			delete log_;
		}
	}

private:
	detail::context* context_;
	CUmodule program_;
	char* log_;
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
		const char * build_options=NULL)
{
	std::ifstream in(filename, std::ios::in);
	AURA_CHECK_ERROR(in);
	in.seekg(0, std::ios::end);
	std::string data;
	data.resize(in.tellg());
	in.seekg(0, std::ios::beg);
	in.read(&data[0], data.size());
	in.close();
	return module(data.c_str(), d, build_options);
}


/**
 * @brief build a kernel module from a string
 *
 * @param str Kernel string
 * @param device device the module is built for
 * @param build_options options for the compiler (optional)
 *
 * @return module reference to compiled module
 */
inline module create_module_from_string(const char* str, device & d,
		const char * build_options=NULL)
{
	return module(str, d, build_options);
}


/**
 * @brief print the module build log
 *
 * @param m the module that is built
 * @param d the device the module is built for
 */

inline void print_module_build_log(const module & m, const device & d)
{
	if (nullptr != m.get_build_log())
	{
		printf("%s\n", m.get_build_log());
	}
}


} // cuda
} // backend_detail
} // aura
} // boost


#endif // AURA_BACKEND_CUDA_MODULE_HPP

