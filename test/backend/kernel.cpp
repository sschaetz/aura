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
"}";

#elif AURA_BACKEND_OPENCL

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
  }
}


