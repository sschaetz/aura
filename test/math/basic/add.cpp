#define BOOST_TEST_MODULE math.add

#include <vector>
#include <stdio.h>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/math/basic/add.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;

// add_float
// _____________________________________________________________________________
typedef std::complex<float> cfloat;


std::default_random_engine generator(1);
std::uniform_real_distribution<float> distribution(-1e5,1e5);
auto random_float = [&](){ return distribution(generator);};
auto random_cfloat = [&](){ return cfloat(random_float(),random_float());};

BOOST_AUTO_TEST_CASE(add_float) 
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


BOOST_AUTO_TEST_CASE(add_cfloat) 
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
		
		std::generate(input1.begin(), input1.end(), random_cfloat);
		std::generate(input2.begin(), input2.end(), random_cfloat);
		
		
		device_array<cfloat> device_input1(y, d);
		device_array<cfloat> device_input2(y, d);
		device_array<cfloat> device_output(y, d);

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
					[](const cfloat& a, const cfloat& b) {
						return a+b;
					}
				);

			
		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				input2.begin())
			);
	}
	
	
}

BOOST_AUTO_TEST_CASE(add_float_cfloat) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};		

	for (auto y : sizes) {
		std::vector<float> input1(y);		
		std::vector<cfloat> input2(y);
		std::vector<cfloat> input1h(y);   // used for checking on the host
		std::vector<cfloat> output(y, cfloat(0.0,0.0));
		
		std::generate(input1.begin(), input1.end(), random_float);
		std::generate(input2.begin(), input2.end(), random_cfloat);
		
		std::transform(input1.begin(), input1.end(),
					input1h.begin(), input1h.begin(),
					[](const float& a, const cfloat& b) {
						return cfloat(a,0);
					}
				);

		device_array<float> device_input1(y, d);
		device_array<cfloat> device_input2(y, d);
		device_array<cfloat> device_output(y, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(output, device_output, f);

		math::add(device_input1, device_input2, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input1h.begin(), input1h.end(),
					input2.begin(), input2.begin(),
					[](const cfloat& a, const cfloat& b) {
						return a+b;
					}
				);
		
			
		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				input2.begin())
			);
	}
	
	
}


BOOST_AUTO_TEST_CASE(add_cfloat_float) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  

	std::vector<int> sizes = {1,2,3,4,5,128,1024,1024*1024,1024*1024*16};	
	

	for (auto y : sizes) {
		std::vector<cfloat> input1(y);
		std::vector<float> input2(y);
		std::vector<cfloat> input2h(y);  // used for checking on the host
		std::vector<cfloat> output(y, cfloat(0.0,0.0));
		
		std::generate(input1.begin(), input1.end(), random_cfloat);
		std::generate(input2.begin(), input2.end(), random_float);
		
		std::transform(input2.begin(), input2.end(),
					input2h.begin(), input2h.begin(),
					[](const float& a, const cfloat& b) {
						return cfloat(a,0);
					}
				);

		device_array<cfloat> device_input1(y, d);
		device_array<float> device_input2(y, d);
		device_array<cfloat> device_output(y, d);

		feed f(d);
		copy(input1, device_input1, f);
		copy(input2, device_input2, f);
		copy(output, device_output, f);

		math::add(device_input1, device_input2, device_output, f);

		copy(device_output, output, f);
		wait_for(f);

		// simulate result on host
		std::transform(input1.begin(), input1.end(),
					input2h.begin(), input2h.begin(),
					[](const cfloat& a, const cfloat& b) {
						return a+b;
					}
				);
		
			
		BOOST_CHECK(
			std::equal(output.begin(), output.end(),
				input2h.begin())
			);
	}
	
	
}
