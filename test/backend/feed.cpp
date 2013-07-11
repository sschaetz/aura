#define BOOST_TEST_MODULE backend.feed

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>

using namespace aura::backend;

// basic
// check if memory can be allocated and used across multiple feeds
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
  int testsize = 512; 
  init();
  int num = device_get_count();
  if(1 < num) {
    device d0(0);  
    feed f0(d0);
    device d1(1);  
    feed f1(d1);

    memory m0 = device_malloc(testsize*sizeof(float), f0);
    memory m1 = device_malloc(testsize*sizeof(float), f1);

    device_free(m0, f0);
    device_free(m1, f1);

  }
}

