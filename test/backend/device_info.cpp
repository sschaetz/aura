#define BOOST_TEST_MODULE backend.device_info

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>

using namespace aura::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
  initialize();
  print_device_info();
}

// extended 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(extended) 
{
  initialize();
  device d(0);
  device_info di = device_get_info(d);
  print_device_info(di); 
}

