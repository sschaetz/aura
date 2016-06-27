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
        /// Create empty library.
        inline explicit library()
                : device_(nullptr)
        {
        }

        /// Prevent copies.
        library(const library&) = delete;
        void operator=(const library&) = delete;

        /// Create library from string.
        inline explicit library(const std::string& kernelstring, device& d,
                const std::string& options = "")
                : device_(&d)
        {
                create_from_string(kernelstring, d, options);
        }

        /// Create library from file.
        inline explicit library(
                boost::aura::path p, device& d, const std::string& options = "")
                : device_(&d)
        {
                auto kernelstring = boost::aura::read_all(p);
                create_from_string(kernelstring, d, options);
        }


        /// Access device.
        const device& get_device()
        {
                return *device_;
        }

        ~library()
        {
                finalize();
        }

private:
        /// Create a library from a string.
        void create_from_string(const std::string& kernelstring, device& d,
                const std::string& opt)
        {
                // Prepend AURA define to kernel.
                auto kernelstring_with_define =
                        std::string("#define AURA_BASE_CUDA\n") + kernelstring;
                d.activate();

                // Create and compile.
                nvrtcProgram program;
                AURA_CUDA_NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,
                        kernelstring_with_define.c_str(), NULL, 0, NULL, NULL));
                AURA_CUDA_NVRTC_SAFE_CALL(
                        nvrtcCompileProgram(program, 0, NULL));

                // Optain PTX.
                size_t ptx_size;
                AURA_CUDA_NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &ptx_size));
                char* ptx = new char[ptx_size];
                AURA_CUDA_NVRTC_SAFE_CALL(nvrtcGetPTX(program, ptx));

                // Store the log.
                size_t log_size;
                AURA_CUDA_NVRTC_SAFE_CALL(
                        nvrtcGetProgramLogSize(program, &log_size));
                log_.resize(log_size);
                AURA_CUDA_NVRTC_SAFE_CALL(
                        nvrtcGetProgramLog(program, &(log_[0])));
                std::cout << log_ << std::endl;

                // Program not needed any more.
                AURA_CUDA_NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

                // Build for device by setting context and JIT argument.
                const std::size_t num_options = 1;
                CUjit_option options[num_options];
                void* values[num_options];

                // Set jit target from context (which we got by setting the
                // device).
                options[0] = CU_JIT_TARGET_FROM_CUCONTEXT;
                values[0] = NULL;

                AURA_CUDA_SAFE_CALL(cuModuleLoadDataEx(
                        &library_, ptx, num_options, options, values));

                d.deactivate();
        }

        /// Finalize object.
        void finalize()
        {
                if (device_ != nullptr)
                {
                        AURA_CUDA_SAFE_CALL(cuModuleUnload(library_));
                }
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
