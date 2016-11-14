#pragma once

#include <boost/aura/base/alang.hpp>
#include <boost/aura/base/opencl/alang.hpp>
#include <boost/aura/base/opencl/device.hpp>
#include <boost/aura/base/opencl/safecall.hpp>
#include <boost/aura/io.hpp>

#include <iostream>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace opencl
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
                create_from_string(kernelstring, options);
        }

        /// Create library from file.
        inline explicit library(
                boost::aura::path p, device& d, const std::string& options = "")
                : device_(&d)
        {
                auto kernelstring = boost::aura::read_all(p);
                create_from_string(kernelstring, options);
        }


        /// Access device.
        const device& get_device() { return *device_; }

        /// Access library.
        cl_program get_base_library() { return library_; }

        const cl_program get_base_library() const { return library_; }

        ~library() { finalize(); }

private:
        /// Create a library from a string.
        void create_from_string(
                const std::string& kernelstring, const std::string& opt)
        {
                shared_alang_header salh;
                alang_header alh;

                // Prepend AURA define to kernel.
                auto kernelstring_with_define =
                        std::string("#define AURA_BASE_OPENCL\n") + salh.get() +
                        std::string("\n") + alh.get() + std::string("\n") +
                        kernelstring;

                int errorcode = 0;
                std::size_t len = kernelstring_with_define.length();
                const char* strings = kernelstring_with_define.c_str();
                library_ =
                        clCreateProgramWithSource(device_->get_base_context(),
                                1, &strings, &len, &errorcode);
                try
                {
                        AURA_OPENCL_CHECK_ERROR(errorcode);
                        AURA_OPENCL_SAFE_CALL(clBuildProgram(library_, 1,
                                &device_->get_base_device(), opt.c_str(), NULL,
                                NULL));
                }
                catch (...)
                {
                        size_t log_size;
                        AURA_OPENCL_SAFE_CALL(clGetProgramBuildInfo(library_,
                                device_->get_base_device(),
                                CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
                        log_.resize(log_size);


                        AURA_OPENCL_SAFE_CALL(clGetProgramBuildInfo(library_,
                                device_->get_base_device(),
                                CL_PROGRAM_BUILD_LOG, log_size, &(log_[0]),
                                NULL));

                        std::cout << log_ << std::endl;

                        throw;
                }
                size_t log_size;
                AURA_OPENCL_SAFE_CALL(clGetProgramBuildInfo(library_,
                        device_->get_base_device(), CL_PROGRAM_BUILD_LOG, 0,
                        NULL, &log_size));
                log_.resize(log_size);


                AURA_OPENCL_SAFE_CALL(clGetProgramBuildInfo(library_,
                        device_->get_base_device(), CL_PROGRAM_BUILD_LOG,
                        log_size, &(log_[0]), NULL));

                std::cout << log_ << std::endl;
        }

        /// Finalize object.
        void finalize()
        {
                if (device_ != nullptr)
                {
                        AURA_OPENCL_SAFE_CALL(clReleaseProgram(library_));
                }
        }

        /// Pointer to device the feed was created for
        device* device_;

        /// Library
        cl_program library_;

        /// Library compile log
        std::string log_;
};


} // opencl
} // base_detail
} // aura
} // boost
