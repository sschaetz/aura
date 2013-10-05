#ifndef AURA_BACKEND_OPENCL_INVOKE_HPP
#define AURA_BACKEND_OPENCL_INVOKE_HPP

#include <assert.h>
#include <cstddef>
#include <CL/cl.h>
#include <aura/detail/svec.hpp>
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

#define AURA_KERNEL_THREAD_LAYOUT_CUDA
void invoke_impl(kernel & k, const grid & g, const block & b, 
  const args_t & a, feed & f) {
  // set parameters
  for(std::size_t i=0; i<a.size(); i++) {
    AURA_OPENCL_SAFE_CALL(clSetKernelArg(k, i, a[i].second, a[i].first));
  }
#ifdef AURA_KERNEL_THREAD_LAYOUT_CUDA
  svec<std::size_t, AURA_MAX_GRID_DIMS+AURA_MAX_BLOCK_DIMS> g_, b_;
  for(std::size_t i=0; i<g.size(); i++) {
    g_.push_back(g[i]);
    b_.push_back(1);
  }
  for(std::size_t i=g.size(); i<g.size()+b.size(); i++) {
    g_.push_back(b[i-g.size()]);
    b_.push_back(b[i-g.size()]);
  }

  // call kernel
  AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(
    f.get_stream(), k, g_.size(), NULL, &g_[0], &b_[0], 0, NULL, NULL));
#else
  assert(g.size() == b.size());
  // call kernel
  AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(
    f.get_stream(), k, g.size(), NULL, &g[0], &b[0], 0, NULL, NULL)); 
#endif
} 

} // namespace detail


// without args
void invoke(kernel & k, const grid & g, const block & b, feed & f) {
  detail::invoke_impl(k, g, b, args_t(), f);
}

// with args
void invoke(kernel & k, const grid & g, const block & b,
  const args_t & a, feed & f) {
  detail::invoke_impl(k, g, b, a, f);
}

} // namespace aura
} // namespace backend_detail
} // namespace opencl

#endif // AURA_BACKEND_OPENCL_INVOKE_HPP

