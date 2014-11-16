#define BOOST_TEST_MODULE device_buffer 

#include <boost/test/unit_test.hpp>
#include <boost/aura/device_buffer.hpp>

using namespace boost::aura::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);
		boost::aura::device_buffer<int> dbi(40, d);
	}
}


