
#define BOOST_TEST_MODULE math.reduced_sum

#include <vector>
#include <stdio.h>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/special/reduced_sum.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/math/complex.hpp>

using namespace boost::aura;
using namespace boost::aura::math;


// conj_float 
// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(reduce_sum) 
{

	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(1);  
	
	std::default_random_engine generator(1);
	std::uniform_real_distribution<float> distribution(-1e5,1e5);
	auto random_float = [&]() -> float { return distribution(generator);};

    std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024*16};
    std::vector<int> channels = {1,10,12};

    // for complex variable:
	for (auto y : sizes) {
		std::cout << "Y:" << y << std::endl;
		for (auto ch : channels) {
			std::vector<cfloat> input(y*ch);
			std::generate(input.begin(), input.end(), random_float);
			std::vector<cfloat> output(y, cfloat(0.0,0.0));

			device_array<cfloat> device_input(y*ch, d);
			device_array<cfloat> device_output(y, d);

			feed f(d);
			copy(input, device_input, f);
			copy(output, device_output, f);

			math::reduced_sum(device_input, device_output, f);

			copy(device_output, output, f);
			wait_for(f);

			// simulate result on host
			std::vector<cfloat> sum(y, cfloat(0.0, 0.0));
			for (int y1 =0; y1 < y; y1++ ) {
				for (int i=0; i < ch; i++) {
					sum[y1] += input[y1+i*y];
				}
			}
			
			BOOST_CHECK_MESSAGE(
				std::equal(output.begin(), output.end(),
					sum.begin()), "(y,ch):" << y << ":" << ch
			);
		}
	}

}
