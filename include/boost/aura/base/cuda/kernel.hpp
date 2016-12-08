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
        /// Create empty object.
        inline explicit kernel()
                : initialized_(false)
        {}

        /// Create kernel from library.
        inline explicit kernel(const std::string& name, library& l)
                : initialized_(true)
        {
                l.get_device().activate();
                AURA_CUDA_SAFE_CALL(cuModuleGetFunction(
                        &kernel_, l.get_base_library(), name.c_str()));
                l.get_device().deactivate();
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
                        kernel_ = nullptr;
                        initialized_ = false;
                }
        }

        /// Destroy kernel.
        inline ~kernel()
        {
                reset();
        }

        /// Access kernel (base).
        CUfunction get_base_kernel() { return kernel_; }

private:
        /// Initialized flag
        bool initialized_;

        /// Kernel handle.
        CUfunction kernel_;
};

} // namespace cuda
} // namespace base_detail
} // namespace aura
} // namespace boost
