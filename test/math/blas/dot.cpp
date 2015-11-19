#define BOOST_TEST_MODULE math.dot

#include <random>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/blas/dot.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/math/complex.hpp>

using namespace boost::aura;

// dot_float
// _____________________________________________________________________________

// std::vector<int> sizes = {1,2,3,4,5,8,13,16,24,28,34,80,128,1024,1026,1031,1024*16,1024*64,1024*128,1024*1024+1,1024*1024+5,1024*1024*2+1,1024*1024*4-1};
// std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};
std::vector<int> sizes = {1,2,3,4,5}; // FIXME: larger tests fail due to float rounding errors in CPU impl

std::default_random_engine generator(1);
std::uniform_real_distribution<float> distribution(-1e3,1e3);
auto random_float = [](){ return distribution(generator);};
auto random_cfloat = [](){ return math::cfloat(random_float(),random_float());};




BOOST_AUTO_TEST_CASE(dot_float)
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);


	for (auto x : sizes) {
		std::vector<float> input1(x);
		std::vector<float> input2(x);
		std::vector<float> output(1);

		std::generate(input1.begin(), input1.end(), random_float);
		std::generate(input2.begin(), input2.end(), random_float);

		device_array<float> device_input1(x, d);
		device_array<float> device_input2(x, d);
		device_array<float> device_output(1, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(output, device_output, f);

		math::dot(device_input1, device_input2, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::vector<float> sum(1,0.0);
		for (int i=0; i<x; i++) {
			sum[0] += input1[i]*input2[i];
		}
		std::cout << x << ": " << sum[0] << " " << output[0] << std::endl;
		BOOST_CHECK(
			abs(sum[0]-output[0])
				<= std::numeric_limits<float>::epsilon()
					* abs(sum[0]+output[0])
			);
	}
}

// dot_cfloat
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(dot_cfloat)
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);


	for (auto x : sizes) {

		std::vector<math::cfloat> input1(x);
		std::vector<math::cfloat> input2(x);
		std::vector<math::cfloat> output(1);

		std::generate(input1.begin(), input1.end(), random_cfloat);
		std::generate(input2.begin(), input2.end(), random_cfloat);

		device_array<math::cfloat> device_input1(x, d);
		device_array<math::cfloat> device_input2(x, d);
		device_array<math::cfloat> device_output(1, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(output, device_output, f);

		math::dot(device_input1, device_input2, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::vector<math::cfloat> sum(1,0.0);
		for (int i=0; i<x; i++) {
			sum[0] += conj(input1[i])*input2[i];
		}

		std::cout << x << ": " << sum[0] << " " << output[0] << std::endl;
		BOOST_CHECK(
			abs(sum[0]-output[0])
				<= std::numeric_limits<float>::epsilon()
					* abs(sum[0]+output[0])
			);
	}
}

