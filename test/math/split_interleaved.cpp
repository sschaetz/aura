#define BOOST_TEST_MODULE math.split_interleaved

#include <vector>
#include <stdio.h>
#include <algorithm>
#include <random>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/split_interleaved.hpp>
#include <boost/aura/math/complex.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;
using namespace boost::aura::math;


// split_interleaved
// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(split_interleaved_float)
{

	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::default_random_engine generator(1);
	std::uniform_real_distribution<float> distribution(-1e5,1e5);
	auto random_float = [&]() -> float { return distribution(generator);};

	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};
	for (auto x : sizes) {
		std::vector<float>  input1(x);
		std::vector<float>  input2(x);
		std::vector<float> output1(x);
		std::vector<float> output2(x);
		std::vector<cfloat> output(x);

		std::generate(input1.begin(), input1.end(), random_float);
		std::generate(input2.begin(), input2.end(), random_float);


		device_array<float> device_input1(x, d);
		device_array<float> device_input2(x, d);
		device_array<cfloat> device_tmpoutput(x, d);
		device_array<float> device_output1(x, d);
		device_array<float> device_output2(x, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);

		math::s2i(device_input1, device_input2, device_tmpoutput, f);
		math::i2s(device_tmpoutput, device_output1, device_output2, f);

		copy(device_output1, output1, f);
		copy(device_output2, output2, f);
		wait_for(f);

		BOOST_CHECK(
			std::equal(output1.begin(), output1.end(),
				input1.begin())
			&&
			std::equal(output2.begin(), output2.end(),
				input2.begin())
			);
	}
}

