#define BOOST_TEST_MODULE backend.feed

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>

using namespace aura::backend;

// pin_unpin 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(pin_unpin) {
  initialize();
  int num = device_get_count();
  if(1 < num) {
    device d0(0);  
    feed f0(d0);
    device d1(1);  
    feed f1(d1);

    d0.pin();
    f0.unset();
#if AURA_BACKEND_CUDA
    CUdevice dev;
    AURA_CUDA_SAFE_CALL(cuCtxGetDevice(&dev));
    BOOST_CHECK(dev == d0.get_device());
#endif
    d0.unpin();
  }
}

// multiple 
// check if memory can be allocated across multiple feeds
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(multiple) {
  int testsize = 512; 
  initialize();
  int num = device_get_count();
  if(1 < num) {
    device d0(0);  
    feed f0(d0);
    device d1(1);  
    feed f1(d1);

    memory m0 = device_malloc(testsize*sizeof(float), d0);
    memory m1 = device_malloc(testsize*sizeof(float), d1);

    device_free(m0, d0);
    device_free(m1, d1);

  }
}

