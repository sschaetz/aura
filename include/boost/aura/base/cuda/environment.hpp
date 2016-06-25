#pragma once

#include <boost/aura/base/cuda/safecall.hpp>

#include <cuda.h>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace cuda
{

/// Initialize base.
inline void initialize()
{
        AURA_CUDA_SAFE_CALL(cuInit(0));
}

/// Finalize base.
inline void finalize()
{
        // Pass
}

} // cuda
} // base_detail
} // aura
} // boost
