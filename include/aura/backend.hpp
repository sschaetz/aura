
#ifndef AURA_BACKEND_HPP 
#define AURA_BACKEND_HPP

#include <aura/config.hpp>

#if AURA_BACKEND_CUDA

#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>
#include <aura/backend/cuda/init.hpp>
#include <aura/backend/cuda/memory.hpp>
#include <aura/backend/cuda/feed.hpp>

#elif AURA_BACKEND_OPENCL

#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>
#include <aura/backend/opencl/init.hpp>
#include <aura/backend/opencl/memory.hpp>
#include <aura/backend/opencl/feed.hpp>

#endif

namespace aura {

namespace backend = backend_detail::AURA_BACKEND_LC;
 
}

#endif // AURA_BACKEND_HPP

