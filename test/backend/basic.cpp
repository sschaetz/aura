#define BOOST_TEST_MODULE backend.basic 

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>

using namespace aura::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic)
{
  int num = device_get_count();
  if(0 < num) {
    device d = device_create(0);  
    context c = context_create(d);
    stream s = stream_create(d, c);
    stream_destroy(s);
    context_destroy(c);
  }
}

// memorypingpong 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(memorypingpong)
{
  int num = device_get_count();
  if(0 < num) {
    device d = device_create(0);  
    context c = context_create(d);
    stream s = stream_create(d, c);
    int testsize = 512; 
    std::vector<float> a1(testsize, 42.);
    std::vector<float> a2(testsize);
    
    memory m = device_malloc(testsize*sizeof(float), c);
    copy(m, &a1[0], testsize*sizeof(float), s); 
    copy(&a2[0], m, testsize*sizeof(float), s);
    stream_synchronize(s);
    device_free(m);
    stream_destroy(s);
    context_destroy(c);
    printf("%f\n", a2[0]);
    if(std::equal(a1.begin(), a1.end(), a2.begin())) {
      printf("Copy test ok.\n");
    } else {
      fprintf(stderr, "Copy test not ok.\n");
    }
  }
}
