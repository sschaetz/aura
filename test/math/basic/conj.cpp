
#define BOOST_TEST_MODULE math.conj

#include <vector>
#include <stdio.h>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/basic/conj.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/math/complex.hpp>

using namespace boost::aura;
using namespace boost::aura::math;


// conj_float 
// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(conj_float) 
{

	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::default_random_engine generator(1);
	std::uniform_real_distribution<float> distribution(-1e5,1e5);
	auto random_float = [&]() -> float { return distribution(generator);};
	
	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};

        
        // for complex variable:
        
	for (auto y : sizes) {
		std::vector<cfloat> input(y);
		std::generate(input.begin(), input.end(), random_float);
		std::vector<cfloat> output(y, cfloat(0.0,0.0));
		std::vector<cfloat> temp(y, cfloat(0.0,0.0));

		device_array<cfloat> device_input(y, d);
 		device_array<cfloat> device_output(y, d);

		feed f(d);
		copy(input, device_input, f);
		copy(output, device_output, f);

		math::conj(device_input, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input.begin(), input.end(),
					temp.begin(),
					[&](const cfloat& a){
						return std::conj(a); 
					}
				);


		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				temp.begin())
		);
	}

}
