#define BOOST_TEST_MODULE backend.fft

#include <complex>
#include <type_traits>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/config.hpp>

#include "fft_data.hpp"

using namespace boost::aura;
using namespace boost::aura::backend;


// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	initialize();
	fft_initialize(); 
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);

	device d(0);
	feed f(d); 

	// 1d
	{
		bounds b(std::extent<decltype(signal_1d_4)>::value);
		std::vector<cfloat> i(product(b), cfloat(0., 0.));
		std::vector<cfloat> o(product(b), cfloat(0., 0.));

		device_array<cfloat> id(b, d);
		device_array<cfloat> od(b, d);

		fft fh(d, f, b, fft::type::c2c);
		copy(signal_1d_4, id.begin(), product(b), f);
		fft_forward(id, od, f);
		copy(od.begin(), &o[0], product(b), f);
		wait_for(f);
		BOOST_CHECK(std::equal(output.begin(), 
					output.end(), spectrum_1d_4));
	}
	
	fft_terminate();
}
# if 0
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(batched_1d) 
{
	int batchsize = 16;
	initialize();
	fft_initialize(); 
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	assert(samples == sizeof(signal) / sizeof(signal[0]));

	std::vector<cfloat> input(samples*batchsize, cfloat(0.0));
	std::vector<cfloat> output(samples*batchsize, cfloat(0.0));
	for(int i=0; i<batchsize; i++) {
		std::copy(&signal[0], &signal[samples], 
				input.begin()+i*samples);
	}
	device d(0);
	feed f(d); 

	device_ptr<cfloat> m1 = 
		device_malloc<cfloat>(batchsize*samples, d);
	device_ptr<cfloat> m2 = 
		device_malloc<cfloat>(batchsize*samples, d);
	copy(m1, &input[0], samples*batchsize, f);
	copy(m2, &output[0], samples*batchsize, f);

	fft fh(d, f, bounds(samples), fft::type::c2c, batchsize);
	fft_forward(m1, m2, fh, f);

	copy(&output[0], m2, samples*batchsize, f);
	wait_for(f);
	for(int i=0; i<batchsize; i++) {
		BOOST_CHECK(std::equal(output.begin()+i*samples, 
					output.begin()+(i+1)*samples, 
					spectrum));
	}	
	device_free(m1);
	device_free(m2);
	fft_terminate();
}
#endif
