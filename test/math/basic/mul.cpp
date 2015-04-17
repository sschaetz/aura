#define BOOST_TEST_MODULE math.mul

#include <vector>
#include <stdio.h>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/basic/mul.hpp>
#include <boost/aura/device_array.hpp>

#include <iomanip>

using namespace boost::aura;

// mul_float
// _____________________________________________________________________________

typedef std::complex<float> cfloat;

std::default_random_engine generator(1);
std::uniform_real_distribution<float> distribution(-1e5,1e5);
auto random_float = [&](){ return distribution(generator);};
auto random_cfloat = [&](){ return cfloat(random_float(),random_float());};

BOOST_AUTO_TEST_CASE(mul_float) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};

	for (auto x : sizes) {
		std::vector<float> input1(x);
		std::vector<float> input2(x);
		std::vector<float> output(x, 0.0);
		
		std::generate(input1.begin(), input1.end(), random_float);
		std::generate(input2.begin(), input2.end(), random_float);

		device_array<float> device_input1(x, d);
		device_array<float> device_input2(x, d);
		device_array<float> device_output(x, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(output, device_output, f);

		math::mul(device_input1, device_input2, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input1.begin(), input1.end(),
					input2.begin(), input2.begin(),
					[](const float& a, const float& b) {
						return a*b;
					}
				);
		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				input2.begin())
			);
	}
}


BOOST_AUTO_TEST_CASE(mul_cfloat) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};	

	for (auto y : sizes) {
		std::vector<cfloat> input1(y);
		std::vector<cfloat> input2(y);
		std::vector<cfloat> output(y, cfloat(0.0,0.0));			

		std::vector<bool> check(y,true);
		std::vector<bool> check1(y,true);
		
		std::generate(input1.begin(), input1.end(), random_cfloat);
		std::generate(input2.begin(), input2.end(), random_cfloat);		
		
		device_array<cfloat> device_input1(y, d);
		device_array<cfloat> device_input2(y, d);
		device_array<cfloat> device_output(y, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(output, device_output, f);

		math::mul(device_input1, device_input2, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input1.begin(), input1.end(),
					input2.begin(), input2.begin(),
					[](const cfloat& a, const cfloat& b) {
						return a*b;
					}
				);
			
		std::transform(output.begin(), output.end(),
					input2.begin(), check.begin(),
					[](const cfloat& a, const cfloat& b) {
						return abs(a-b) <= std::numeric_limits<float>::epsilon() * abs(a+b);   //see c++ documentation about epsilon (numeric limits) 
					}
				);
						
		BOOST_CHECK(
			std::equal(check.begin(), check.end(),
				check1.begin())
			);
			
	}
	
}

BOOST_AUTO_TEST_CASE(mul_float_cfloat) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};
	for (auto z : sizes) {
		std::vector<float> input1(z);
		std::vector<cfloat> input1h(z);   // used for checking on the host
		std::vector<cfloat> input2(z);
		std::vector<cfloat> output(z);
		
		std::generate(input1.begin(), input1.end(), random_float);
		std::generate(input2.begin(), input2.end(), random_cfloat);
		
		std::transform(input1.begin(), input1.end(),
					input1h.begin(), input1h.begin(),
					[](const float& a, const cfloat& b) {
						return cfloat(a,0);
					}
				);

		device_array<float> device_input1(z, d);
		device_array<cfloat> device_input2(z, d);
		device_array<cfloat> device_output(z, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(output, device_output, f);

		math::mul(device_input1, device_input2, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input1h.begin(), input1h.end(),
					input2.begin(), input2.begin(),
					[](const cfloat& a, const cfloat& b) {
						return a*b;
					}
				);
		
		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				input2.begin())
			);
	}
	
	
}


BOOST_AUTO_TEST_CASE(mul_cfloat_float) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};	
	

	for (auto m : sizes) {
		std::vector<cfloat> input1(m);
		std::vector<float> input2(m);
		std::vector<cfloat> input2h(m);  // used for checking on the host
		std::vector<cfloat> output(m, cfloat(0.0,0.0));
		
		std::generate(input1.begin(), input1.end(), random_cfloat);
		std::generate(input2.begin(), input2.end(), random_float);
		
		std::transform(input2.begin(), input2.end(),
					input2h.begin(), input2h.begin(),
					[](const float& a, const cfloat& b) {
						return cfloat(a,0);
					}
				);

		device_array<cfloat> device_input1(m, d);
		device_array<float> device_input2(m, d);
		device_array<cfloat> device_output(m, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(output, device_output, f);

		math::mul(device_input1, device_input2, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input1.begin(), input1.end(),
					input2h.begin(), input2h.begin(),
					[](const cfloat& a, const cfloat& b) {
						return a*b;
					}
				);
		
		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				input2h.begin())
			);
	}
	
	
}
