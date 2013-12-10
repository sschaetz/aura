#ifndef AURA_BACKEND_CUDA_INVOKE_HPP
#define AURA_BACKEND_CUDA_INVOKE_HPP

#include <cstddef>
#include <cuda.h>
#include <aura/backend/cuda/kernel.hpp>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/feed.hpp>
#include <aura/backend/cuda/mesh.hpp>
#include <aura/backend/cuda/bundle.hpp>
#include <aura/backend/cuda/args.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

namespace detail {

void invoke_impl(kernel & k, const mesh & m, const bundle & b, 
  const args_t & a, feed & f) {
  // handling for non 3-dimensional mesh and bundle sizes
  std::size_t meshx = m[0], meshy = 1, meshz = 1;
  std::size_t bundlex = b[0], bundley = 1, bundlez = 1;
  if(m.size() > 1) {
    meshy = m[1];
  }
  if(m.size() > 2) {
    meshz = m[2];
  }
  if(b.size() > 1) {
    bundley = b[1];
  }
  if(b.size() > 2) {
    bundlez = b[2];
  }

  // number of bundles subdivides meshes but CUDA has a
  // "consists of" semantic so we need less mesh elements
  meshx /= bundlex;
  meshy /= bundley;
  meshz /= bundlez;

  f.set();
  AURA_CUDA_SAFE_CALL(cuLaunchKernel(k, meshx, meshy, meshz, 
    bundlex, bundley, bundlez, 0, f.get_backend_stream(), 
    const_cast<void**>(&a.second[0]), NULL)); 
  f.unset();
  free(a.first);
}

} // namespace detail

/// invoke kernel without args
void invoke(kernel & k, const mesh & m, const bundle & b, feed & f) {
  detail::invoke_impl(k, m, b, args_t(), f);
}

/// invoke kernel with args
void invoke(kernel & k, const mesh & m, const bundle & b,
  const args_t & a, feed & f) {
  detail::invoke_impl(k, m, b, a, f);
}

} // namespace aura
} // namespace backend_detail
} // namespace cuda

#endif // AURA_BACKEND_CUDA_INVOKE_HPP

