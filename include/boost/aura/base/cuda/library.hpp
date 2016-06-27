#pragma once

#include <boost/aura/base/cuda/device.hpp>
#include <boost/aura/base/cuda/safecall.hpp>
#include <boost/aura/io.hpp>

#include <cuda.h>
#include <nvrtc.h>

#include <iostream>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace cuda
{


class library
{
public:
        // Create empty library.
        inline explicit library()
        {
        }

        // Create library from string.
        inline explicit library(const std::string& kernelstring, device& d,
                const std::string& options = "")
        {
                create_from_string(kernelstring, d, options);
        }

        // Create library from file.
        inline explicit library(boost::aura::path p, device& d,
                const std::string& options = "")
        {
                auto kernelstring = boost::aura::read_all(p);
                create_from_string(kernelstring, d, options);
        }


        // Access device.
        const device& get_device()
        {
                return *device_;
        }

private:
        void create_from_string(const std::string& kernelstring, device& d,
                const std::string& opt)
        {
                // Prepend AURA define to kernel.
                auto kernelstring_with_define =
                        std::string("#define AURA_BASE_CUDA\n") + kernelstring;
                d.activate();

		// Create and compile.
		nvrtcProgram program;
		nvrtcCreateProgram(&program,
                                kernelstring_with_define.c_str() ,
                                NULL, 0, NULL, NULL);
		nvrtcCompileProgram(program, 0, NULL);

		// Optain PTX.
		size_t ptx_size;
		nvrtcGetPTXSize(program, &ptx_size);
		char* ptx = new char[ptx_size];
		nvrtcGetPTX(program, ptx);

		// Store the log.
		size_t log_size;
		nvrtcGetProgramLogSize(program, &log_size);
                log_.resize(log_size);
		nvrtcGetProgramLog(program, &(log_[0]));
                std::cout << log_ << std::endl;

                // Program not needed any more.
		nvrtcDestroyProgram(&program);

		// Build for device by setting context and JIT argument.
		const std::size_t num_options = 1;
		CUjit_option options[num_options];
		void * values[num_options];

		// Set jit target from context (which we got by setting the device).
		options[0] = CU_JIT_TARGET_FROM_CUCONTEXT;
		values[0] = NULL;

		AURA_CUDA_SAFE_CALL(cuModuleLoadDataEx(&library_, ptx, num_options,
                        options, values));

                d.deactivate();
        }

        /// Finalize object.
        void finalize()
        {
        }

        /// Pointer to device the feed was created for
        device* device_;

        /// Library
        CUmodule library_;

        /// Library compile log
        std::string log_;
};


} // cuda
} // base_detail
} // aura
} // boost
