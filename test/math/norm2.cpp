#define BOOST_TEST_MODULE math.norm2 

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/norm2.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;

// norm2_float
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(norm2_float) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<float> input(128, 42.0);
	std::vector<float> output(1, 0.0);
	device_array<float> device_input(128, d);
	device_array<float> device_output(1, d);

	feed f(d);
	copy(input, device_input, f);
	math::norm2(device_input, device_output, f);

	copy(device_output, output, f);
	wait_for(f);

	BOOST_CHECK(std::accumulate(input.begin(), input.end(), 0.0f) ==
			output[0]);

}

