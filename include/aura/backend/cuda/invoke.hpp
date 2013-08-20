#ifndef AURA_BACKEND_CUDA_INVOKE_HPP
#define AURA_BACKEND_CUDA_INVOKE_HPP

#include <cstddef>
#include <CL/cl.h>
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

template <std::size_t N0, std::size_t N1, std::size_t N2>
void invoke_impl(kernel & k, typename grid_t<N0>::type g, 
  typename block_t<N1>::type b, typename args_t<N2>::type a, feed & f) {
  
  // set parameters
  for(std::size_t i=0; i<N2; i++) {
    AURA_CUDA_SAFE_CALL(clSetKernelArg(k, i, a[i].second, a[i].first));
  }
#ifdef AURA_KERNEL_THREAD_LAYOUT_OPENCL
  std::array<std::size_t, N0+N1> g_, b_;
  for(std::size_t i=0; i<N0; i++) {
    g_[i] = g[i];
    b_[i] = 1;
  }
  for(std::size_t i=N0; i<N0+N1; i++) {
    g_[i] = b[i-N0];
    b_[i] = b[i-N0];
  }

  // call kernel
  AURA_CUDA_SAFE_CALL(clEnqueueNDRangeKernel(
    f.get_stream(), k, g_.size(), NULL, &g_[0], &b_[0], 0, NULL, NULL)); 
#else
  assert(g.size() == b.size());
  // call kernel
  AURA_CUDA_SAFE_CALL(clEnqueueNDRangeKernel(
    f.get_stream(), k, g.size(), NULL, &g[0], &b[0], 0, NULL, NULL)); 
#endif
}

} // namespace detail

// maybe this can also be generalized for CUDA and OpenCL

// no args
void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b, feed & f) {
  detail::invoke_impl<1, 1, 0>(k, g,   b, args_t<0>::type(), f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b, feed & f) {
  detail::invoke_impl<1, 2, 0>(k, g,   b, args_t<0>::type(), f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b, feed & f) {
  detail::invoke_impl<1, 3, 0>(k, g,   b, args_t<0>::type(), f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b, feed & f) {
  detail::invoke_impl<2, 1, 0>(k, g,   b, args_t<0>::type(), f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b, feed & f) {
  detail::invoke_impl<2, 2, 0>(k, g,   b, args_t<0>::type(), f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b, feed & f) {
  detail::invoke_impl<2, 3, 0>(k, g,   b, args_t<0>::type(), f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b, feed & f) {
  detail::invoke_impl<3, 1, 0>(k, g,   b, args_t<0>::type(), f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b, feed & f) {
  detail::invoke_impl<3, 2, 0>(k, g,   b, args_t<0>::type(), f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b, feed & f) {
  detail::invoke_impl<3, 3, 0>(k, g,   b, args_t<0>::type(), f);
}


// with args
void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<1, 1, 1>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<2>::type a, feed & f) {
  detail::invoke_impl<1, 1, 2>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<3>::type a, feed & f) {
  detail::invoke_impl<1, 1, 3>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<4>::type a, feed & f) {
  detail::invoke_impl<1, 1, 4>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<5>::type a, feed & f) {
  detail::invoke_impl<1, 1, 5>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<6>::type a, feed & f) {
  detail::invoke_impl<1, 1, 6>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<7>::type a, feed & f) {
  detail::invoke_impl<1, 1, 7>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<8>::type a, feed & f) {
  detail::invoke_impl<1, 1, 8>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<9>::type a, feed & f) {
  detail::invoke_impl<1, 1, 9>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<1>::type b,
  typename args_t<10>::type a, feed & f) {
  detail::invoke_impl<1, 1, 10>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<1, 2, 1>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<2>::type a, feed & f) {
  detail::invoke_impl<1, 2, 2>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<3>::type a, feed & f) {
  detail::invoke_impl<1, 2, 3>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<4>::type a, feed & f) {
  detail::invoke_impl<1, 2, 4>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<5>::type a, feed & f) {
  detail::invoke_impl<1, 2, 5>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<6>::type a, feed & f) {
  detail::invoke_impl<1, 2, 6>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<7>::type a, feed & f) {
  detail::invoke_impl<1, 2, 7>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<8>::type a, feed & f) {
  detail::invoke_impl<1, 2, 8>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<9>::type a, feed & f) {
  detail::invoke_impl<1, 2, 9>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<2>::type b,
  typename args_t<10>::type a, feed & f) {
  detail::invoke_impl<1, 2, 10>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<1, 3, 1>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<2>::type a, feed & f) {
  detail::invoke_impl<1, 3, 2>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<3>::type a, feed & f) {
  detail::invoke_impl<1, 3, 3>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<4>::type a, feed & f) {
  detail::invoke_impl<1, 3, 4>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<5>::type a, feed & f) {
  detail::invoke_impl<1, 3, 5>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<6>::type a, feed & f) {
  detail::invoke_impl<1, 3, 6>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<7>::type a, feed & f) {
  detail::invoke_impl<1, 3, 7>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<8>::type a, feed & f) {
  detail::invoke_impl<1, 3, 8>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<9>::type a, feed & f) {
  detail::invoke_impl<1, 3, 9>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<1>::type g,
  typename block_t<3>::type b,
  typename args_t<10>::type a, feed & f) {
  detail::invoke_impl<1, 3, 10>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<2, 1, 1>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<2>::type a, feed & f) {
  detail::invoke_impl<2, 1, 2>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<3>::type a, feed & f) {
  detail::invoke_impl<2, 1, 3>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<4>::type a, feed & f) {
  detail::invoke_impl<2, 1, 4>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<5>::type a, feed & f) {
  detail::invoke_impl<2, 1, 5>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<6>::type a, feed & f) {
  detail::invoke_impl<2, 1, 6>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<7>::type a, feed & f) {
  detail::invoke_impl<2, 1, 7>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<8>::type a, feed & f) {
  detail::invoke_impl<2, 1, 8>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<9>::type a, feed & f) {
  detail::invoke_impl<2, 1, 9>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<1>::type b,
  typename args_t<10>::type a, feed & f) {
  detail::invoke_impl<2, 1, 10>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<2, 2, 1>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<2>::type a, feed & f) {
  detail::invoke_impl<2, 2, 2>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<3>::type a, feed & f) {
  detail::invoke_impl<2, 2, 3>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<4>::type a, feed & f) {
  detail::invoke_impl<2, 2, 4>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<5>::type a, feed & f) {
  detail::invoke_impl<2, 2, 5>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<6>::type a, feed & f) {
  detail::invoke_impl<2, 2, 6>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<7>::type a, feed & f) {
  detail::invoke_impl<2, 2, 7>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<8>::type a, feed & f) {
  detail::invoke_impl<2, 2, 8>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<9>::type a, feed & f) {
  detail::invoke_impl<2, 2, 9>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<2>::type b,
  typename args_t<10>::type a, feed & f) {
  detail::invoke_impl<2, 2, 10>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<2, 3, 1>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<2>::type a, feed & f) {
  detail::invoke_impl<2, 3, 2>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<3>::type a, feed & f) {
  detail::invoke_impl<2, 3, 3>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<4>::type a, feed & f) {
  detail::invoke_impl<2, 3, 4>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<5>::type a, feed & f) {
  detail::invoke_impl<2, 3, 5>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<6>::type a, feed & f) {
  detail::invoke_impl<2, 3, 6>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<7>::type a, feed & f) {
  detail::invoke_impl<2, 3, 7>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<8>::type a, feed & f) {
  detail::invoke_impl<2, 3, 8>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<9>::type a, feed & f) {
  detail::invoke_impl<2, 3, 9>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<2>::type g,
  typename block_t<3>::type b,
  typename args_t<10>::type a, feed & f) {
  detail::invoke_impl<2, 3, 10>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<3, 1, 1>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<2>::type a, feed & f) {
  detail::invoke_impl<3, 1, 2>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<3>::type a, feed & f) {
  detail::invoke_impl<3, 1, 3>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<4>::type a, feed & f) {
  detail::invoke_impl<3, 1, 4>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<5>::type a, feed & f) {
  detail::invoke_impl<3, 1, 5>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<6>::type a, feed & f) {
  detail::invoke_impl<3, 1, 6>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<7>::type a, feed & f) {
  detail::invoke_impl<3, 1, 7>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<8>::type a, feed & f) {
  detail::invoke_impl<3, 1, 8>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<9>::type a, feed & f) {
  detail::invoke_impl<3, 1, 9>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<1>::type b,
  typename args_t<10>::type a, feed & f) {
  detail::invoke_impl<3, 1, 10>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<3, 2, 1>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<2>::type a, feed & f) {
  detail::invoke_impl<3, 2, 2>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<3>::type a, feed & f) {
  detail::invoke_impl<3, 2, 3>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<4>::type a, feed & f) {
  detail::invoke_impl<3, 2, 4>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<5>::type a, feed & f) {
  detail::invoke_impl<3, 2, 5>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<6>::type a, feed & f) {
  detail::invoke_impl<3, 2, 6>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<7>::type a, feed & f) {
  detail::invoke_impl<3, 2, 7>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<8>::type a, feed & f) {
  detail::invoke_impl<3, 2, 8>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<9>::type a, feed & f) {
  detail::invoke_impl<3, 2, 9>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<2>::type b,
  typename args_t<10>::type a, feed & f) {
  detail::invoke_impl<3, 2, 10>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<3, 3, 1>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<2>::type a, feed & f) {
  detail::invoke_impl<3, 3, 2>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<3>::type a, feed & f) {
  detail::invoke_impl<3, 3, 3>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<4>::type a, feed & f) {
  detail::invoke_impl<3, 3, 4>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<5>::type a, feed & f) {
  detail::invoke_impl<3, 3, 5>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<6>::type a, feed & f) {
  detail::invoke_impl<3, 3, 6>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<7>::type a, feed & f) {
  detail::invoke_impl<3, 3, 7>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<8>::type a, feed & f) {
  detail::invoke_impl<3, 3, 8>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<9>::type a, feed & f) {
  detail::invoke_impl<3, 3, 9>(k, g, b, a, f);
}

void invoke(kernel & k, typename grid_t<3>::type g,
  typename block_t<3>::type b,
  typename args_t<10>::type a, feed & f) {
  detail::invoke_impl<3, 3, 10>(k, g, b, a, f);
}

} // namespace aura
} // namespace backend_detail
} // namespace cuda

#endif // AURA_BACKEND_CUDA_INVOKE_HPP
