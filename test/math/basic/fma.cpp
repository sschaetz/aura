
#define BOOST_TEST_MODULE math.fma

#include <vector>
#include <stdio.h>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/basic/fma.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/math/complex.hpp>

using namespace boost::aura;
using namespace boost::aura::math;


// fma_float 
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(fma_float) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};
	for (auto x : sizes) {
		std::vector<float> input1(x, 1.0);
		std::vector<float> input2(x, 1.0);
		std::vector<float> input3(x, 1.0);
		std::vector<float> output(x, 0.0);

		device_array<float> device_input1(x, d);
		device_array<float> device_input2(x, d);
		device_array<float> device_input3(x, d);
// 		device_array<float> device_output(x, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(input3, device_input3, f);


		math::fma(device_input1, device_input2,
			  device_input3, f);

		copy(device_input3, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input1.begin(), input1.end(),
					input2.begin(), input2.begin(),
					[&](const float& a, const float& b) {
						return a*b;
					}
				);
		std::transform(input2.begin(), input2.end(),
					input3.begin(), input3.begin(),
					[&](const float& a, const float& b) {
						return a+b;
					}
                    );
		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				input3.begin())
			);
	}

        
        // for complex variable:
        
        for (auto y : sizes) {
		std::vector<cfloat> input1(y, cfloat(1.0,41.0));
		std::vector<cfloat> input2(y, cfloat(1.0,41.0));
		std::vector<cfloat> input3(y, cfloat(1.0,41.0));
		std::vector<cfloat> output(y, cfloat(0.0,0.0));

		device_array<cfloat> device_input1(y, d);
		device_array<cfloat> device_input2(y, d);
		device_array<cfloat> device_input3(y, d);
// 		device_array<cfloat> device_output(y, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(input3, device_input3, f);

		math::fma(device_input1, device_input2,
			  device_input3, f);

		copy(device_input3, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input1.begin(), input1.end(),
					input2.begin(), input2.begin(),
					[&](const cfloat& a, const cfloat& b) {
						return a*b;
					}
				);

		std::transform(input2.begin(), input2.end(),
					input3.begin(), input3.begin(),
					[&](const cfloat& a, const cfloat& b) {
						return a+b;
					}
                                );

		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				input3.begin())
		);
	}
}



