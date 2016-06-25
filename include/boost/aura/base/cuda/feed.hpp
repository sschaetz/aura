#pragma once

#include <boost/aura/base/cuda/device.hpp>
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


class feed
{
public:
        /// create empty feed object without device and stream
        inline explicit feed()
                : device_(nullptr)
        {
        }

        /**
         * Create device feed for device.
         *
         * @param d device to create feed for
         */
        inline explicit feed(device &d)
                : device_(&d)
        {
                device_->activate();
                AURA_CUDA_SAFE_CALL(
                        cuStreamCreate(&feed_, 0 /*CU_STREAM_NON_BLOCKING*/));
                device_->deactivate();
        }

        /**
         * move constructor, move feed information here, invalidate other
         *
         * @param f feed to move here
         */
        feed(feed &&f)
                : device_(f.device_)
                , feed_(f.feed_)
        {
                f.device_ = nullptr;
        }

        /**
         * move assignment, move feed information here, invalidate other
         *
         * @param f feed to move here
         */
        feed &operator=(feed &&f)
        {
                finalize();
                device_ = f.device_;
                feed_ = f.feed_;
                f.device_ = nullptr;
                return *this;
        }

        /// Destroy feed.
        inline ~feed()
        {
                finalize();
        }

        /// Wait until all commands in the feed have finished.
        inline void synchronize()
        {
                device_->activate();
                AURA_CUDA_SAFE_CALL(cuStreamSynchronize(feed_));
                device_->deactivate();
        }

        /// @copydoc boost::aura::base::cuda::device::get_base_device()
        inline const CUdevice &get_base_device() const
        {
                return device_->get_base_device();
        }

        /// @copydoc boost::aura::base::cuda::device::get_base_contet()
        inline const CUcontext &get_base_context() const
        {
                return device_->get_base_context();
        }

        /// Access base feed.
        inline const CUstream &get_base_feed() const
        {
                return feed_;
        }

        /// Access base feed.
        inline CUstream &get_base_feed()
        {
                return feed_;
        }

        // Access device.
        const device &get_device()
        {
                return *device_;
        }

private:
        /// Finalize object.
        void finalize()
        {
                if (nullptr != device_)
                {
                        device_->activate();
                        AURA_CUDA_SAFE_CALL(cuStreamDestroy(feed_));
                        device_->deactivate();
                }
        }

        /// Pointer to device the feed was created for
        device *device_;

        /// Stream handle
        CUstream feed_;
};

/**
 * @brief wait for a feed to finish all operations
 *
 * @param f the feed to wait for
 */
inline void wait_for(feed &f)
{
        f.synchronize();
}

} // cuda
} // base_detail
} // aura
} // boost
