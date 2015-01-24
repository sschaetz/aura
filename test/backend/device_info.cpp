#define BOOST_TEST_MODULE backend.device_info

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>

using namespace boost::aura::backend;

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
  for (int i=0; i<device_get_count(); i++) {
	  device d(i);
	  device_info di = device_get_info(d);
	  std::cout << "device ordinal " << i << ": ";
	  print_device_info(di);
  }
}

