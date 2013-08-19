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
  // call kernel
  AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(
    f.get_stream(), k, g.size(), NULL, &g[0], &b[0], 0, NULL, NULL)); 
}

} // namespace detail

// we should maybe generate 3x3x10 of this function
void invoke(kernel & k, typename grid_t<1>::type g, 
  typename block_t<1>::type b, typename args_t<1>::type a, feed & f) {
  detail::invoke_impl<1, 1, 1>(k, g, b, a, f);
}
 

} // opencl
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_INVOKE_HPP

