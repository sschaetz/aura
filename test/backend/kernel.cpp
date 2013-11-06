#define BOOST_TEST_MODULE backend.kernel

// these kernels should be called with the CUDA style thread layout
#define AURA_KERNEL_THREAD_LAYOUT_CUDA

#include <cstring>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>
#include <aura/config.hpp>

using namespace aura::backend;

#if AURA_BACKEND_OPENCL

const char * kernel_file = "test/kernel.cl"; 

#elif AURA_BACKEND_CUDA

const char * kernel_file = "test/kernel.ptx"; 

#endif

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
  initialize();
  int num = device_get_count();
  if(0 < num) {
    device d(0); 
    module m = create_module_from_file(kernel_file, d);
    kernel k = create_kernel(m, "noarg");
    (void)k; 
  }
}

// basic

// invoke_
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(invoke_) {
  initialize();
  int num = device_get_count();
  if(0 < num) {
    device d(0);  
    feed f(d);
    std::size_t xdim = 16;
    std::size_t ydim = 16;
    
    std::vector<float> a1(xdim*ydim, 41.);
    std::vector<float> a2(xdim*ydim);
    
    module mod = create_module_from_file(kernel_file, d);
    kernel k = create_kernel(mod, "simple_add"); 
    memory mem = device_malloc(xdim*ydim*sizeof(float), d);
    
    copy(mem, &a1[0], xdim*ydim*sizeof(float), f); 
    invoke(k, mesh(ydim), bundle(xdim), args(mem), f);
    copy(&a2[0], mem, xdim*ydim*sizeof(float), f);
    wait_for(f);

    for(std::size_t i=0; i<a1.size(); i++) {
      a1[i] += 1.0;
    }
    BOOST_CHECK(std::equal(a1.begin(), a1.end(), a2.begin()));
    device_free(mem, d);
  }
}

// invoke_noarg
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(invoke_noarg) {
  initialize();
  int num = device_get_count();
  if(0 < num) {
    device d(0);  
    feed f(d);
    std::size_t xdim = 16;
    std::size_t ydim = 16;
    
    module mod = create_module_from_file(kernel_file, d);
    kernel k = create_kernel(mod, "noarg"); 
    invoke(k, mesh(ydim), bundle(xdim), f);
    wait_for(f);
  }
}
