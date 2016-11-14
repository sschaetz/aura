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
        /// Create kernel from library.
        inline explicit kernel(const std::string& name, library& l)
        {
                int errorcode = 0;
                kernel_ = clCreateKernel(
                        l.get_base_library(), name.c_str(), &errorcode);
                AURA_OPENCL_CHECK_ERROR(errorcode);
        }

        /// Prevent copies.
        kernel(const kernel&) = delete;
        void operator=(const kernel&) = delete;

        /// Destroy kernel.
        inline ~kernel() { AURA_OPENCL_SAFE_CALL(clReleaseKernel(kernel_)); }

        /// Access kernel (base).
        cl_kernel get_base_kernel() { return kernel_; }

private:
        cl_kernel kernel_;
};

} // namespace opencl
} // namespace base_detail
} // namespace aura
} // namespace boost
