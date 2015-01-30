#define BOOST_TEST_MODULE device_lock 

#include <iostream>
#include <memory>

#include <boost/test/unit_test.hpp>
#include <boost/aura/device_lock.hpp>

namespace ip = boost::interprocess;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	auto l = boost::aura::create_device_lock(0);
	if (l) {
		std::cout << "got lock!" << std::endl;
	} else {
		std::cout << "could not get lock!" << std::endl;
	}
}



