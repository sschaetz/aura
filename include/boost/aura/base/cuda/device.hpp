#pragma once

#include <boost/aura/base/check_initialized.hpp>
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

class device
{
public:
        /// Query the number of devices in the system.
        static std::size_t num()
        {
                int num_devices;
                AURA_CUDA_SAFE_CALL(cuDeviceGetCount(&num_devices));
                return (std::size_t)num_devices;
        }

public:
        /// Create empty device.
        inline explicit device()
                : initialized_(false)
                , ordinal_(-1)
        {
        }

        /// Create device form ordinal.
        /// @param ordinal Device number
        inline explicit device(std::size_t ordinal)
                : initialized_(false)
                , ordinal_(ordinal)
        {
                AURA_CUDA_SAFE_CALL(cuDeviceGet(&device_, ordinal));
                AURA_CUDA_SAFE_CALL(cuCtxCreate(&context_, 0, device_));
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
                , context_(other.context_)
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
                context_ = other.context_;

                other.initialized_ = false;
                other.ordinal_ = -1;
                return *this;
        }

        /// Reset.
        inline void reset()
        {
                if (initialized_)
                {
                        AURA_CUDA_SAFE_CALL(cuCtxDestroy(context_));
                        initialized_ = false;
                }
                ordinal_ = -1;
        }

        /// Destroy device.
        inline ~device() { reset(); }

        /// Access the device handle.
        inline const CUdevice& get_base_device() const
        {
                AURA_CHECK_INITIALIZED(initialized_);
                return device_;
        }

        /// Access the context handle.
        inline const CUcontext& get_base_context() const
        {
                AURA_CHECK_INITIALIZED(initialized_);
                return context_;
        }

        /// Access the device ordinal.
        inline std::size_t get_ordinal() const
        {
                AURA_CHECK_INITIALIZED(initialized_);
                return ordinal_;
        }

        /// Make device activate device.
        inline void activate() const
        {
                AURA_CHECK_INITIALIZED(initialized_);
                AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(context_));
        }

        /// Undo make device active device.
        inline void deactivate() const
        {
                AURA_CHECK_INITIALIZED(initialized_);
                AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(NULL));
        }

        /// Query initialized state.
        inline bool initialized() const { return initialized_; }

private:
        /// Initialized flag
        bool initialized_;

        /// Device ordinal
        std::size_t ordinal_;

        /// Device handle
        CUdevice device_;

        /// Context handle
        CUcontext context_;
};

} // namespace cuda
} // namespace base_detail
} // namespace aura
} // namespace boost
