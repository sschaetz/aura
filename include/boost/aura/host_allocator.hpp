
#ifndef AURA_HOST_ALLOCATOR_HPP
#define AURA_HOST_ALLOCATOR_HPP

#include <boost/aura/config.hpp>

#if defined AURA_BACKEND_CUDA
#elif defined AURA_BACKEND_OPENCL
	#include <boost/aura/backend/opencl/detail/host_allocator.hpp>
#endif

namespace boost
{
namespace aura 
{

#ifdef AURA_BACKEND_CUDA
namespace backend = backend_detail::cuda;
#elif AURA_BACKEND_OPENCL
namespace backend = backend_detail::opencl;
#endif

using backend::detail::host_allocator;

} // namespace aura
} // namespace boost

#endif // AURA_HOST_ALLOCATOR_HPP

