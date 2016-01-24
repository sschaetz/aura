#define BOOST_TEST_MODULE backend.kernel

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
	module m = create_module_from_file(kernel_file, d,
	AURA_BACKEND_COMPILE_FLAGS);
	kernel k = create_kernel(m, "donothing");
	(void)k;
}
