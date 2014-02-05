#define BOOST_TEST_MODULE device_array

#include <boost/test/unit_test.hpp>
#include <aura/device_array.hpp>

using namespace aura;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);
		device_array<int> array1(40, d);
		device_array<int> array2(bounds(40, 20, 10), d);
		BOOST_CHECK(40 == array1.size());
		BOOST_CHECK(40*20*10 == array2.size());
	}
}


