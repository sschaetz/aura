#pragma once

#include <boost/aura/base/opencl/library.hpp>
#include <boost/aura/base/opencl/safecall.hpp>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace opencl
{

class kernel
{
public:
        /// @copydoc boost::aura::base::cuda::kernel()
        inline explicit kernel()
        {}

        /// @copydoc boost::aura::base::cuda::kernel(const std::string& name, library& l)
        inline explicit kernel(const std::string& name, library& l)
        {
                int errorcode = 0;
                kernel_ = clCreateKernel(
                        l.get_base_library(), name.c_str(), &errorcode);
                AURA_OPENCL_CHECK_ERROR(errorcode);
                initialized_ = true;
        }

        /// Prevent copies.
        kernel(const kernel&) = delete;
        void operator=(const kernel&) = delete;

        /// Move construct.
        kernel(kernel&& other)
                : initialized_(other.initialized_)
                , kernel_(other.kernel_)
        {
                other.initialized_ = false;
        }

        /// Move assign.
        kernel& operator=(kernel&& other)
        {
                reset();

                initialized_ = other.initialized_;
                kernel_ = other.kernel_;

                other.initialized_ = false;
                return *this;
        }

        /// Reset.
        inline void reset()
        {
                if (initialized_)
                {
                        AURA_OPENCL_SAFE_CALL(clReleaseKernel(kernel_));
                        initialized_ = false;
                }
        }

        /// Destroy kernel.
        inline ~kernel()
        {
                reset();
        }

        /// Access kernel (base).
        cl_kernel get_base_kernel() { return kernel_; }

private:
        /// Initialized flag
        bool initialized_ { false };

        /// Kernel handle.
        cl_kernel kernel_;
};

} // namespace opencl
} // namespace base_detail
} // namespace aura
} // namespace boost
