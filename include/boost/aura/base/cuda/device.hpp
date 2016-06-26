#pragma once

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
        /**
         * Create device form ordinal.
         *
         * @param ordinal Device number
         */
        inline explicit device(std::size_t ordinal)
                : ordinal_(ordinal)
        {
                AURA_CUDA_SAFE_CALL(cuDeviceGet(&device_, ordinal));
                AURA_CUDA_SAFE_CALL(cuCtxCreate(&context_, 0, device_));
        }

        /// Destroy device.
        inline ~device()
        {
                AURA_CUDA_SAFE_CALL(cuCtxDestroy(context_));
        }

        /// Access the device handle.
        inline const CUdevice& get_base_device() const
        {
                return device_;
        }

        /// Access the context handle.
        inline const CUcontext& get_base_context() const
        {
                return context_;
        }

        /// Access the device ordinal.
        inline std::size_t get_ordinal() const
        {
                return ordinal_;
        }

        /// Make device activate device.
        inline void activate() const
        {
                AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(context_));
        }

        /// Undo make device active device.
        inline void deactivate() const
        {
                AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(NULL));
        }

private:
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
