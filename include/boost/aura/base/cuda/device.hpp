#pragma once

#include <boost/aura/base/cuda/safecall.hpp>

#include <cuda.h>

namespace boost {
namespace aura {
namespace base_detail {
namespace cuda {

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

        /**
         * Destroy device.
         */
        inline ~device()
        {
                AURA_CUDA_SAFE_CALL(cuCtxDestroy(context_));
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
