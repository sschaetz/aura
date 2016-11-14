#pragma once

#include <boost/aura/base/cuda/library.hpp>
#include <boost/aura/base/cuda/safecall.hpp>

#include <cuda.h>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace cuda
{

class kernel
{
public:
        /// Create kernel from library.
        inline explicit kernel(const std::string& name, library& l)
        {
                l.get_device().activate();
                AURA_CUDA_SAFE_CALL(cuModuleGetFunction(
                        &kernel_, l.get_base_library(), name.c_str()));
                l.get_device().deactivate();
        }

        /// Prevent copies.
        kernel(const kernel&) = delete;
        void operator=(const kernel&) = delete;

        /// Destroy kernel.
        inline ~kernel() {}

        /// Access kernel (base).
        CUfunction get_base_kernel() { return kernel_; }

private:
        CUfunction kernel_;
};

} // namespace cuda
} // namespace base_detail
} // namespace aura
} // namespace boost
