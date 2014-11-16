#define BOOST_TEST_MODULE device_buffer 

#include <boost/test/unit_test.hpp>
#include <boost/aura/device_buffer.hpp>

using namespace boost::aura;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);
		print_device_info(device_get_info(d));
	}
}

