#define BOOST_TEST_MODULE backend.host_memory

#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>

using namespace aura::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	device d(0);
	float* ptr1 = (float*)host_malloc(5*sizeof(float));
	host_free((void*)ptr1);
	float * ptr2 = host_malloc<float>(5);
	host_free(ptr2);
}

