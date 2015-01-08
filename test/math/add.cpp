#define BOOST_TEST_MODULE math.add

#include <vector>
#include <stdio.h>
#include <algorithm>
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

	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};
	for (auto x : sizes) {
		std::vector<float> input1(x, 41.0);
		std::vector<float> input2(x, 1.0);
		std::vector<float> output(x, 0.0);

		device_array<float> device_input1(x, d);
		device_array<float> device_input2(x, d);
		device_array<float> device_output(x, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(output, device_output, f);

		math::add(device_input1, device_input2, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input1.begin(), input1.end(),
					input2.begin(), input2.begin(),
					[](const float& a, const float& b) {
						return a+b;
					}
				);
		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				input2.begin())
			);
	}
}

