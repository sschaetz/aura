#pragma once

#include <boost/aura/base/metal/library.hpp>
#include <boost/aura/base/metal/safecall.hpp>

#if ! __has_feature(objc_arc)
#error This file must be compiled with ARC. Either turn on ARC for the project or use -fobjc-arc flag
#endif

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace metal
{

class kernel
{
public:
        /// @copydoc boost::aura::base::cuda::kernel()
        inline explicit kernel() {}

        /// @copydoc boost::aura::base::cuda::kernel(const std::string& name,
        /// library& l)
        inline explicit kernel(const std::string& name, library& l)
        {
            @autoreleasepool {
                NSString* kernel_name = @(name.c_str());
                kernel_ =
                        [l.get_base_library() newFunctionWithName:kernel_name];
                AURA_METAL_CHECK_ERROR(kernel_);
                initialized_ = true;
            }
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
                        kernel_ = nil;
                        initialized_ = false;
                }
        }

        /// Destroy kernel.
        inline ~kernel() { reset(); }

        /// Access kernel (base).
        id<MTLFunction> get_base_kernel() { return kernel_; }


private:
        /// Initialized flag
        bool initialized_{false};

        /// Kernel handle.
        id<MTLFunction> kernel_;
};

} // namespace metal
} // namespace base_detail
} // namespace aura
} // namespace boost
