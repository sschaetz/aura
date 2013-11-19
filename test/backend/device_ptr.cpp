#define BOOST_TEST_MODULE backend.basic 

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>

using namespace aura::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
  initialize();
  int num = device_get_count();
  if(0 < num) {
    device d(0);  
    feed f(d);
    memory m = device_malloc(15*sizeof(float), d);
    device_ptr<float> ptr1(m, d);
    device_ptr<float> ptr2(m, d);
    BOOST_CHECK(ptr1 == ptr2);
    ptr1 = ptr1+1;
    BOOST_CHECK(ptr1 != ptr2);
    ++ptr2;
    BOOST_CHECK(ptr1 == ptr2);
    device_free(m, d);
  }
}

