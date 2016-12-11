#pragma once

#include <boost/aura/base/check_initialized.hpp>
#include <boost/aura/base/metal/safecall.hpp>

#include <TargetConditionals.h>
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
#if TARGET_IPHONE_SIMULATOR == 1
                return 0;
#elif TARGET_OS_IPHONE == 1
                return 1;
#elif TARGET_OS_MAC == 1
                return [MTLCopyAllDevices() count];
#endif
        }

public:
        /// @copydoc boost::aura::base::cuda::device::device()
        inline explicit device()
                : initialized_(false)
                , ordinal_(-1)
        {
        }

        /// @copydoc boost::aura::base::cuda::device::device(std::size_t)
        inline explicit device(std::size_t ordinal)
                : initialized_(false)
                , ordinal_(ordinal)
        {
                device_ = MTLCreateSystemDefaultDevice();
                AURA_METAL_CHECK_ERROR(device_);
                initialized_ = true;
        }

        /// Prevent copies.
        device(const device&) = delete;
        void operator=(const device&) = delete;

        /// Move construct.
        device(device&& other)
                : initialized_(other.initialized_)
                , ordinal_(other.ordinal_)
                , device_(other.device_)
        {
                other.initialized_ = false;
                other.ordinal_ = -1;
        }

        /// Move assign.
        device& operator=(device&& other)
        {
                reset();

                initialized_ = other.initialized_;
                ordinal_ = other.ordinal_;
                device_ = other.device_;

                other.initialized_ = false;
                other.ordinal_ = -1;
                return *this;
        }

        // Reset.
        inline void reset()
        {
                if (initialized_)
                {
                        device_ = nil;
                        initialized_ = false;
                }
                ordinal_ = -1;
        }

        /// @copydoc boost::aura::base::cuda::device::~device()
        inline ~device() { reset(); }

        /// @copydoc boost::aura::base::cuda::device::get_base_device()
        inline __strong id<MTLDevice> get_base_device()
        {
                AURA_CHECK_INITIALIZED(initialized_);
                return device_;
        }

        /// @copydoc boost::aura::base::cuda::device::get_ordinal()
        inline std::size_t get_ordinal() const
        {
                AURA_CHECK_INITIALIZED(initialized_);
                return ordinal_;
        }

        /// @copydoc boost::aura::base::cuda::device::activate()
        inline void activate() const { AURA_CHECK_INITIALIZED(initialized_); }

        /// @copydoc boost::aura::base::cuda::device::deactivate()
        inline void deactivate() const { AURA_CHECK_INITIALIZED(initialized_); }

        /// Query initialized state.
        inline bool initialized() const { return initialized_; }

private:
        /// Initialized flag
        bool initialized_;

        /// Device ordinal
        std::size_t ordinal_;

        /// Device handle
        id<MTLDevice> device_;
};

} // namespace metal
} // namespace base_detail
} // namespace aura
} // namespace boost
