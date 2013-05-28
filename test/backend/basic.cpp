#define BOOST_TEST_MODULE backend.basic 

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

