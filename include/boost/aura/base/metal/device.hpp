#pragma once

#include <boost/aura/base/metal/safecall.hpp>

#import <Metal/Metal.h>

#include <cstddef>

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
        /// Query the number of devices in the system.
        static std::size_t num()
        {
                return [MTLCopyAllDevices() count];
        }

public:
        /// @copydoc boost::aura::base::cuda::device::device()
        inline explicit device(std::size_t ordinal)
                : ordinal_(ordinal)
        {
                device_ = MTLCreateSystemDefaultDevice();
                AURA_METAL_CHECK_ERROR(device_);
        }

        /// Prevent copies.
        device(const device&) = delete;
        void operator=(const device&) = delete;

        /// @copydoc boost::aura::base::cuda::device::~device()
        inline ~device() { device_ = nil; }

        /// @copydoc boost::aura::base::cuda::device::get_base_device()
        inline __strong id<MTLDevice>& get_base_device() { return device_; }

        /// @copydoc boost::aura::base::cuda::device::get_ordinal()
        inline std::size_t get_ordinal() const { return ordinal_; }

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

        inline bool supports_shared_memory() const { return true; }

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
