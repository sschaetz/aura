#ifndef AURA_BACKEND_CUDA_DETAIL_FEED_MARKER_HELPER_HPP
#define AURA_BACKEND_CUDA_DETAIL_FEED_MARKER_HELPER_HPP


#include <boost/aura/backend/cuda/mark.hpp>
#include <boost/aura/backend/cuda/feed.hpp>

namespace boost
{
namespace aura
{
namespace backend_detail
{
namespace cuda 
{
namespace detail
{

inline void set_feed(feed& f) 
{
	f.set();
}

inline void unset_feed(feed& f) 
{
	f.unset();
}

inline const CUstream& get_backend_stream(feed& f)
{
	return f.get_backend_stream();
}

inline CUevent get_event(mark& m)
{
	return m.get_event();
}

} // detail
} // cuda
} // backend_detail
} // aura
} // boost
#endif // AURA_BACKEND_CUDA_DETAIL_FEED_MARKER_HELPER_HPP

