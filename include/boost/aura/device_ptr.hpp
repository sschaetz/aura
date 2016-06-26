#pragma once

#if defined AURA_BASE_CUDA
#include <boost/aura/base/cuda/device_ptr.hpp>
#elif defined AURA_BASE_OPENCL
#include <boost/aura/base/opencl/device_ptr.hpp>
#elif defined AURA_BASE_METAL
#include <boost/aura/base/metal/device_ptr.hpp>
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

using base::device_ptr;
using base::device_malloc;
using base::device_free;

} // namespace aura
} // namespace boost
