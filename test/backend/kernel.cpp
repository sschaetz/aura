#define BOOST_TEST_MODULE backend.kernel

#include <cstring>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>
#include <aura/config.hpp>

using namespace aura::backend;

#ifdef AURA_BACKEND_OPENCL

const char * kernel_code =  
"__kernel void dataParallel(__global float * A,"
"  __global float* B, __global float* C)"
"{"
"    int base = 4*get_global_id(0);"
"    C[base+0] = A[base+0] + B[base+0];"
"    C[base+1] = A[base+1] - B[base+1];"
"    C[base+2] = A[base+2] * B[base+2];"
"    C[base+3] = A[base+3] / B[base+3];"
"}"
"__kernel void simple_add(__global float * A)"
"{"
"    int id = get_global_id(0) * get_global_size(0) + "
"      get_local_id(1);"
"    A[id] += 1.0;"
"}";

#elif AURA_BACKEND_CUDA

const char * kernel_code = "";

#endif

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
  init();
  int num = device_get_count();
  if(0 < num) {
    device d(0); 
    module m = build_module_from_source(kernel_code, strlen(kernel_code), d);
    kernel k = create_kernel(m, "dataParallel");
    (void)k; 
  }
}

// invoke
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(invoke_) {
  init();
  int num = device_get_count();
  if(0 < num) {
    device d(0);  
    feed f(d);
    std::size_t xdim = 16;
    std::size_t ydim = 16;
    
    std::vector<float> a1(xdim*ydim, 41.);
    std::vector<float> a2(xdim*ydim);
    
    module mod = build_module_from_source(kernel_code, strlen(kernel_code), d);
    kernel k = create_kernel(mod, "simple_add"); 
    memory mem = device_malloc(xdim*ydim*sizeof(float), d);
    
    copy(mem, &a1[0], xdim*ydim*sizeof(float), f); 
    invoke(k, grid(ydim), block(xdim), args(mem), f);
    copy(&a2[0], mem, xdim*ydim*sizeof(float), f);
    f.synchronize();
    
    for(std::size_t i=0; i<a1.size(); i++) {
      a1[i] += 1.0;
    }
    BOOST_CHECK(std::equal(a1.begin(), a1.end(), a2.begin()));
    device_free(mem, d);
  }
}
