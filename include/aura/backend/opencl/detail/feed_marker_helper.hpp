#ifndef AURA_BACKEND_OPENCL_DETAIL_FEED_MARKER_HELPER_HPP
#define AURA_BACKEND_OPENCL_DETAIL_FEED_MARKER_HELPER_HPP


#include <aura/backend/opencl/mark.hpp>
#include <aura/backend/opencl/feed.hpp>

namespace aura
{
namespace backend_detail
{
namespace opencl 
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

inline const cl_command_queue& get_backend_stream(feed& f)
{
	return f.get_backend_stream();
}

inline cl_event get_event(mark& m)
{
	return m.get_event();
}

} // detail
} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_DETAIL_FEED_MARKER_HELPER_HPP

