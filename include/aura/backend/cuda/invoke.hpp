#ifndef AURA_BACKEND_CUDA_INVOKE_HPP
#define AURA_BACKEND_CUDA_INVOKE_HPP

#include <cstddef>
#include <cuda.h>
#include <aura/backend/cuda/kernel.hpp>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/feed.hpp>
#include <aura/backend/cuda/grid.hpp>
#include <aura/backend/cuda/block.hpp>
#include <aura/backend/cuda/args.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

namespace detail {

void invoke_impl(kernel & k, const grid & g, const block & b, 
  const args_t & a, feed & f) {
  // handling for non 3-dimensional grid and block sizes
  std::size_t gridx = g[0], gridy = 1, gridz = 1;
  std::size_t blockx = b[0], blocky = 1, blockz = 1;
  if(g.size() > 1) {
    gridy = g[1];
  }
  if(g.size() > 2) {
    gridz = g[2];
  }
  if(b.size() > 1) {
    blocky = b[1];
  }
  if(b.size() > 2) {
    blockz = b[2];
  }

  f.set();
  AURA_CUDA_SAFE_CALL(cuLaunchKernel(k, gridx, gridy, gridz, 
    blockx, blocky, blockz, 0, f.get_stream(), 
    const_cast<void**>(&a[0]), NULL)); 
  f.unset();
}

} // namespace detail

/// invoke kernel without args
void invoke(kernel & k, const grid & g, const block & b, feed & f) {
  detail::invoke_impl(k, g, b, args_t(), f);
}

/// invoke kernel with args
void invoke(kernel & k, const grid & g, const block & b,
  const args_t & a, feed & f) {
  detail::invoke_impl(k, g, b, a, f);
}

} // namespace aura
} // namespace backend_detail
} // namespace cuda

#endif // AURA_BACKEND_CUDA_INVOKE_HPP

