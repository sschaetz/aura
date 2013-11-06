#ifndef AURA_BACKEND_OPENCL_INVOKE_HPP
#define AURA_BACKEND_OPENCL_INVOKE_HPP

#include <assert.h>
#include <cstddef>
#include <CL/cl.h>
#include <aura/detail/svec.hpp>
#include <aura/backend/opencl/kernel.hpp>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/feed.hpp>
#include <aura/backend/opencl/mesh.hpp>
#include <aura/backend/opencl/bundle.hpp>
#include <aura/backend/opencl/args.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

namespace detail {

#define AURA_KERNEL_THREAD_LAYOUT_CUDA
void invoke_impl(kernel & k, const mesh & m, const bundle & b, 
  const args_t & a, feed & f) {
  // set parameters
  for(std::size_t i=0; i<a.size(); i++) {
    AURA_OPENCL_SAFE_CALL(clSetKernelArg(k, i, a[i].second, a[i].first));
  }
#ifdef AURA_KERNEL_THREAD_LAYOUT_CUDA
  svec<std::size_t, AURA_MAX_MESH_DIMS+AURA_MAX_BUNDLE_DIMS> m_, b_;
  for(std::size_t i=0; i<m.size(); i++) {
    m_.push_back(m[i]);
    b_.push_back(1);
  }
  for(std::size_t i=m.size(); i<m.size()+b.size(); i++) {
    m_.push_back(b[i-m.size()]);
    b_.push_back(b[i-m.size()]);
  }

  // call kernel
  AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(
    f.get_backend_stream(), k, m_.size(), NULL, &m_[0], &b_[0], 0, NULL, NULL));
#else
  assert(m.size() == b.size());
  // call kernel
  AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(
    f.get_backend_stream(), k, m.size(), NULL, &m[0], &b[0], 0, NULL, NULL)); 
#endif
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
} // namespace opencl

#endif // AURA_BACKEND_OPENCL_INVOKE_HPP

