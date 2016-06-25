#pragma once

#include <boost/aura/base/metal/safecall.hpp>

#import <Metal/Metal.h>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace metal
{

class device
{
public:
        /// @copydoc boost::aura::base::cuda::device::device()
        inline explicit device(std::size_t ordinal)
                : ordinal_(ordinal)
        {
                device_ = MTLCreateSystemDefaultDevice();
                AURA_METAL_CHECK_ERROR(device_);
        }

        /// @copydoc boost::aura::base::cuda::device::~device()
        inline ~device()
        {
                device_ = nil;
        }

        /// @copydoc boost::aura::base::cuda::device::get_base_device()
        inline __strong id<MTLDevice> &get_base_device()
        {
                return device_;
        }

        /// @copydoc boost::aura::base::cuda::device::get_ordinal()
        inline std::size_t get_ordinal() const
        {
                return ordinal_;
        }

        /// @copydoc boost::aura::base::cuda::device::activate()
        inline void activate() const
        {
                // Pass
        }

        /// @copydoc boost::aura::base::cuda::device::deactivate()
        inline void deactivate() const
        {
                // Pass
        }

private:
        /// Device ordinal
        std::size_t ordinal_;

        /// Device handle
        id<MTLDevice> device_;
};

} // namespace metal
} // namespace base_detail
} // namespace aura
} // namespace boost
