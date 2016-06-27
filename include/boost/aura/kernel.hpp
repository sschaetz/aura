#pragma once

#if defined AURA_BASE_CUDA
#include <boost/aura/base/cuda/kernel.hpp>
#elif defined AURA_BASE_OPENCL
#include <boost/aura/base/opencl/kernel.hpp>
#elif defined AURA_BASE_METAL
#include <boost/aura/base/metal/kernel.hpp>
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

using base::kernel;

} // namespace aura
} // namespace boost
