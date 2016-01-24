#define BOOST_TEST_MODULE backend.kernel

#include <fstream>
#include <cstring>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/config.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;
using namespace boost::aura::backend;

const char * kernel_file = AURA_UNIT_TEST_LOCATION"kernel.cc";

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic)
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(AURA_UNIT_TEST_DEVICE < num);
	device d(AURA_UNIT_TEST_DEVICE);
    std::ifstream s;
    s.open(kernel_file);
    library l(s, d, AURA_BACKEND_COMPILE_FLAGS);
	auto k = l.get_kernel("donothing");
	(void)k;
}


// basic2
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic2)
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(AURA_UNIT_TEST_DEVICE < num);
	device d(AURA_UNIT_TEST_DEVICE);
    library l = make_library_from_file(kernel_file, d,
            AURA_BACKEND_COMPILE_FLAGS);
	auto k = l.get_kernel("donothing");
	(void)k;
}
