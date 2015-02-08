#define BOOST_TEST_MODULE device_lock 

#include <iostream>
#include <memory>

#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>

using namespace boost::aura;

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

// create_device
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(create_device) 
{
	// we use two processes here, one must throw
	if (fork() == 0) {
		initialize();
		device d0= backend::create_device_exclusive();	
		device d1= backend::create_device_exclusive();
		exit(0);
	} else {
		initialize();
		device d0 = backend::create_device_exclusive();	
		device d1 = backend::create_device_exclusive();
		exit(0);
	}
}

