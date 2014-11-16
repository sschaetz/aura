#define BOOST_TEST_MODULE backend.mark

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>

using namespace boost::aura::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);  
		feed f(d);
		feed f2(d);
		mark m;
		insert(f, m);
		wait_for(m);
		f.continue_when(m);
	}
}


