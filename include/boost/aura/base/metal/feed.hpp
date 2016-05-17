#pragma once

#include <boost/aura/base/metal/device.hpp>
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

namespace detail
{

/// Wrapper for command_buffer.
struct command_buffer
{
    id<MTLCommandBuffer> command_buffer;
};

} // detail

class feed
{
public:
        /// @copydoc boost::aura::base::cuda::feed::feed()
	inline explicit feed()
                : device_(nullptr) {}

        /// @copydoc boost::aura::base::cuda::feed::feed(device&)
	inline explicit feed(device& d)
                : device_(&d)
                , feed_([device_->get_base_device() newCommandQueue])
	{
                AURA_METAL_CHECK_ERROR(feed_);
        }

        /// @copydoc boost::aura::base::cuda::feed::feed(feed&&)
	feed(feed&& f)
                : device_(f.device_)
                , feed_(f.feed_)
	{
		f.device_ = nil;
	}

        /// @copydoc boost::aura::base::cuda::feed::operator=()
	feed& operator=(feed&& f)
	{
		finalize();
		device_ = f.device_;
		feed_ = f.feed_;
		f.device_ = nil;
		return *this;
	}

        /// @copydoc boost::aura::base::cuda::feed::~feed()
	inline ~feed()
	{
		finalize();
	}

        /// @copydoc boost::aura::base::cuda::feed::synchronize()
	inline void synchronize()
	{
                if (command_buffers_.empty())
                {
                        return;
                }
                [command_buffers_.back().command_buffer waitUntilCompleted];
                command_buffers_.clear();
	}

        /// @copydoc boost::aura::base::cuda::device::get_base_device()
	inline const __strong id<MTLDevice>& get_base_device() const
	{
		return device_->get_base_device();
	}

        /// @copydoc boost::aura::base::cuda::feed::get_base_feed()
	inline const id<MTLCommandQueue> get_base_feed() const
	{
		return feed_;
	}

        /// @copydoc boost::aura::base::cuda::feed::get_base_feed()
	inline id<MTLCommandQueue> get_base_feed()
	{
		return feed_;
	}

        /// @copydoc boost::aura::base::cuda::feed::get_device()
        const device& get_device()
        {
                return *device_;
        }

        /// Create and return new command buffer.
        /// @note Metal specific.
        /// @return New command buffer.
        detail::command_buffer& get_command_buffer()
        {
                command_buffers_.emplace_back(detail::command_buffer());
                command_buffers_.back().command_buffer =
                        [feed_ commandBuffer];
                AURA_METAL_CHECK_ERROR(command_buffers_.back().command_buffer);
                return command_buffers_.back();
        }

private:
	/// Finalize object.
	void finalize()
	{
		if(nullptr != device_)
                {
		        device_ = nil;
		}
                command_buffers_.clear();
	}

	/// Pointer to device the feed was created for
	device * device_;

        /// Store list of all created command buffers.
        std::list<detail::command_buffer> command_buffers_;

	/// Feed handle.
        id<MTLCommandQueue> feed_;
};

/**
 * @brief wait for a feed to finish all operations
 *
 * @param f the feed to wait for
 */
inline void wait_for(feed& f)
{
	f.synchronize();
}

} // metal
} // base_detail
} // aura
} // boost

