#pragma once

#include <boost/aura/base/metal/library.hpp>
#include <boost/aura/base/metal/safecall.hpp>

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
        /// Create kernel from library.
        inline explicit kernel(const std::string& name, library& l)
        {
                NSString* kernel_name = @(name.c_str());
                kernel_ =
                        [l.get_base_library() newFunctionWithName:kernel_name];
                AURA_METAL_CHECK_ERROR(kernel_);
        }

        /// Prevent copies.
        kernel(const kernel&) = delete;
        void operator=(const kernel&) = delete;

        /// Destroy kernel.
        inline ~kernel() { kernel_ = nil; }

        /// Access kernel (base).
        id<MTLFunction> get_base_kernel() { return kernel_; }


private:
        id<MTLFunction> kernel_;
};

} // namespace metal
} // namespace base_detail
} // namespace aura
} // namespace boost
