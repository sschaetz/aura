#pragma once

#if defined AURA_BASE_CUDA
#include <boost/aura/base/cuda/feed.hpp>
#elif defined AURA_BASE_OPENCL
#include <boost/aura/base/opencl/feed.hpp>
#elif defined AURA_BASE_METAL
#include <boost/aura/base/metal/feed.hpp>
#endif

namespace boost
{
namespace aura
{

#if defined AURA_BASE_CUDA
namespace base = base_detail::cuda;
#elif defined AURA_BASE_OPENCL
namespace base = base_detail::opencl;
#elif defined AURA_BASE_METAL
namespace base = base_detail::metal;
#endif

using base::feed;

/**
 * @brief wait for a feed to finish all operations
 *
 * @param f the feed to wait for
 */
inline void wait_for(feed& f) { f.synchronize(); }

} // namespace aura
} // namespace boost
