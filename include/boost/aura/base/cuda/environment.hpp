#pragma once

#include <boost/aura/base/cuda/safecall.hpp>

#include <cuda.h>

namespace boost {
namespace aura {
namespace base_detail {
namespace cuda {

/// Initialize backend.
inline void initialize()
{
        AURA_CUDA_SAFE_CALL(cuInit(0));
}

/// Finalize backend.
inline void finalize()
{
        // Pass
}

} // cuda
} // base_detail
} // aura
} // boost

