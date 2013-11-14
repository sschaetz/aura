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

void invoke_impl(kernel & k, const mesh & m, const bundle & b, 
  const args_t & a, feed & f) {
  // set parameters
  for(std::size_t i=0; i<a.size(); i++) {
    AURA_OPENCL_SAFE_CALL(clSetKernelArg(k, i, a[i].second, a[i].first));
  }
  // handling for non 3-dimensional mesh and bundle sizes
  mesh tm;
  tm.push_back(m[0]);
  tm.push_back(1);
  tm.push_back(1);
  bundle tb; 
  tb.push_back(1);
  tb.push_back(1);
  tb.push_back(1);

  if(m.size() > 1) {
    tm[1] = m[1];
  }
  if(m.size() > 2) {
    tm[2] = m[2];
  }
  if(b.size() > 0 && m.size() > 0) {
    tb[m.size()-1] = b[b.size()-1];
  }
  if(b.size() > 1 && m.size() > 0) {
    tb[m.size()-2] = b[b.size()-2];
  }
  if(b.size() > 2 && m.size() > 2) {
    tb[m.size()-3] = b[b.size()-3];
  }
  // call kernel
  AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(
    f.get_backend_stream(), k, m.size(), NULL, 
    &tm[0], &tb[0], 0, NULL, NULL)); 
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

