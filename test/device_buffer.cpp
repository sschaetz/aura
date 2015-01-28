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

// resize 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(resize) {
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);
		boost::aura::device_buffer<int> dbi(40, d);
		dbi.resize(60);
		BOOST_CHECK(dbi.size() == 60);
		
		boost::aura::device_buffer<int> dbj;
		dbj.resize(60, d);
		BOOST_CHECK(dbj.size() == 60);
	}
}

