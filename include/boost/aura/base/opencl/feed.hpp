#pragma once

#include <boost/aura/base/opencl/device.hpp>
#include <boost/aura/base/opencl/safecall.hpp>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace opencl
{


class feed
{
public:
        /// Create empty feed object without device.
        inline explicit feed()
                : device_(nullptr)
        {
        }

        /**
         * Create device feed for device.
         *
         * @param d device to create feed for
         */
        inline explicit feed(device& d)
                : device_(&d)
        {
                int errorcode = 0;
                feed_ = clCreateCommandQueue(device_->get_base_context(),
                        device_->get_base_device(), 0, &errorcode);
                AURA_OPENCL_CHECK_ERROR(errorcode);
        }

        /**
         * move constructor, move feed information here, invalidate other
         *
         * @param f feed to move here
         */
        feed(feed&& f)
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
        feed& operator=(feed&& f)
        {
                finalize();
                device_ = f.device_;
                feed_ = f.feed_;
                f.device_ = nullptr;
                return *this;
        }

        /// Destroy feed.
        inline ~feed() { finalize(); }

        /// Wait until all commands in the feed have finished.
        inline void synchronize() { AURA_OPENCL_SAFE_CALL(clFinish(feed_)); }

        /// @copydoc boost::aura::base::cuda::device::get_base_device()
        inline cl_device_id& get_base_device() const
        {
                return device_->get_base_device();
        }

        /// @copydoc boost::aura::base::cuda::device::get_base_contet()
        inline const cl_context& get_base_context() const
        {
                return device_->get_base_context();
        }

        /// Access const base feed.
        inline const cl_command_queue& get_base_feed() const { return feed_; }

        /// Access base feed.
        inline cl_command_queue& get_base_feed() { return feed_; }

        /// @copydoc boost::aura::base::cuda::feed::get_device()
        device& get_device() { return *device_; }

        const device& get_device() const { return *device_; }


private:
        /// Finalize object.
        void finalize()
        {
                if (nullptr != device_)
                {
                        AURA_OPENCL_SAFE_CALL(clReleaseCommandQueue(feed_));
                }
        }

        /// Pointer to device the feed was created for
        device* device_;

        /// Stream handle
        cl_command_queue feed_;
};

} // opencl
} // base_detail
} // aura
} // boost
