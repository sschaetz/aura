#define BOOST_TEST_MODULE math.add

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/add.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;

// add_float
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(add_float) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<float> input1(128, 41.0);
	std::vector<float> input2(128, 1.0);
	std::vector<float> output(128, 0.0);

	device_array<float> device_input1(128, d);
	device_array<float> device_input2(128, d);
	device_array<float> device_output(128, d);

	feed f(d);
	copy(input1, device_input1, f);
	copy(input2, device_input2, f);
	copy(output, device_output, f);

	math::add(device_input1, device_input2, device_output, f);

	copy(device_output, output, f);
	wait_for(f);

	/*
	BOOST_CHECK(std::accumulate(input.begin(), input.end(), 0.0f) ==
			output[0]);
	*/
}

