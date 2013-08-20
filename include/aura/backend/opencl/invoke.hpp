#ifndef AURA_BACKEND_OPENCL_INVOKE_HPP
#define AURA_BACKEND_OPENCL_INVOKE_HPP

#include <cstddef>
#include <CL/cl.h>
#include <aura/backend/opencl/kernel.hpp>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/feed.hpp>
#include <aura/backend/opencl/grid.hpp>
#include <aura/backend/opencl/block.hpp>
#include <aura/backend/opencl/args.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

namespace detail {

template <std::size_t N0, std::size_t N1, std::size_t N2>
void invoke_impl(kernel & k, typename grid_t<N0>::type g, 
  typename block_t<N1>::type b, typename args_t<N2>::type a, feed & f) {
  
  // set parameters
  for(std::size_t i=0; i<N2; i++) {
    AURA_OPENCL_SAFE_CALL(clSetKernelArg(k, i, a[i].second, a[i].first));
  }
  
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
  AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(
    f.get_stream(), k, g_.size(), NULL, &g_[0], &b_[0], 0, NULL, NULL)); 
}

} // namespace detail

// maybe this can also be generalized for CUDA and OpenCL
// what about no args?

// generated using tools/gen_invoker.py
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

} // opencl
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_INVOKE_HPP

