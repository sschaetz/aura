#ifndef AURA_BACKEND_CUDA_FEED_HPP
#define AURA_BACKEND_CUDA_FEED_HPP

#include <boost/move/move.hpp>
#include <cuda.h>
#include <boost/aura/backend/cuda/call.hpp>
#include <boost/aura/backend/cuda/device.hpp>

namespace boost
{
namespace aura
{
namespace backend_detail
{
namespace cuda
{


// trick to avoid circular dependency fallacy issue
// file detail/feed_marker_helper.hpp contains the code
class mark;
class feed;

namespace detail
{

void set_feed(feed& f); 
void unset_feed(feed& f); 
const CUstream& get_backend_stream(feed& f);
CUevent get_event(mark& m);

} // namespace detail


/**
 * feed class
 */
class feed
{

private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(feed)

public:

	/// create empty feed object without device and stream
	inline explicit feed() : context_(nullptr) {}

	/**
	 * create device feed for device
	 *
	 * @param d device to create feed for
	 *
	 * const device & is not allowed since an actual instance is needed
	 */
	inline explicit feed(device& d) : context_(d.get_context())
	{
		context_->set();
		AURA_CUDA_SAFE_CALL(cuStreamCreate(&stream_, 
					0 /*CU_STREAM_NON_BLOCKING*/));
		context_->unset();
	}

	/**
	 * move constructor, move feed information here, invalidate other
	 *
	 * @param f feed to move here
	 */
	feed(BOOST_RV_REF(feed) f) :
		context_(f.context_), stream_(f.stream_)
	{
		f.context_ = nullptr;
	}

	/**
	 * move assignment, move feed information here, invalidate other
	 *
	 * @param f feed to move here
	 */
	feed& operator=(BOOST_RV_REF(feed) f)
	{
		finalize();
		context_ = f.context_;
		stream_ = f.stream_;
		f.context_ = nullptr;
		return *this;
	}

	/// destroy feed
	inline ~feed()
	{
		finalize();
	}

	/// wait until all commands in the feed have finished
	inline void synchronize() const
	{
		context_->set();
		AURA_CUDA_SAFE_CALL(cuStreamSynchronize(stream_));
		context_->unset();
	}

	/// feed should continue if a mark is reached in another stream
	inline void continue_when(mark& m)
	{
		context_->set();
		AURA_CUDA_SAFE_CALL(
		        cuStreamWaitEvent(
		                stream_,
		                detail::get_event(m),
		                0
		        )
		);
		context_->unset();
	}

	/// make feed active
	inline void set() const
	{
		context_->set();
	}

	/// undo make feed active
	inline void unset() const
	{
		context_->unset();
	}

	/// get device
	inline const CUdevice & get_backend_device() const
	{
		return context_->get_backend_device();
	}

	/// get context
	inline const CUcontext & get_backend_context() const
	{
		return context_->get_backend_context();
	}

	/// get stream
	inline const CUstream & get_backend_stream() const
	{
		return stream_;
	}

	/// access the context handle
	inline detail::context * get_context()
	{
		return context_;
	}

private:
	/// finalize object (called from dtor and move assign)
	void finalize()
	{
		if(nullptr != context_) {
			context_->set();
			AURA_CUDA_SAFE_CALL(cuStreamDestroy(stream_));
			context_->unset();
		}
	}

private:
	/// reference to device the feed was created for
	detail::context * context_;
	/// stream handle
	CUstream stream_;
};

/**
 * @brief wait for a feed to finish all operations
 *
 * @param f the feed to wait for
 */
inline void wait_for(feed & f)
{
	f.synchronize();
}

} // cuda
} // backend_detail
} // aura
} // boost

// trick to avoid cirular dependency
#include <boost/aura/backend/cuda/detail/feed_marker_helper.hpp>

#endif // AURA_BACKEND_CUDA_FEED_HPP

