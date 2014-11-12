#define BOOST_TEST_MODULE backend.fft

#include <complex>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>
#include <aura/config.hpp>

typedef std::complex<float> cfloat;

using namespace aura::backend;

const int samples = 4;
const cfloat signal[] = 
      {cfloat(1, 1), cfloat(2, 2), cfloat(3, 3), cfloat(4, 4)};
const cfloat spectrum[] = 
      {cfloat(10, 10), cfloat(-4, 0), cfloat(-2, -2), cfloat(0, -4)};

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	initialize();
	fft_initialize(); 
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	assert(samples == sizeof(signal) / sizeof(signal[0]));

	std::vector<cfloat> input(signal, signal+samples);
	std::vector<cfloat> output(samples, cfloat(555., 666.));
	device d(0);
	feed f(d); 

	device_ptr<cfloat> m1 = device_malloc<cfloat>(samples, d);
	device_ptr<cfloat> m2 = device_malloc<cfloat>(samples, d);
	copy(m1, &input[0], samples, f);
	copy(m2, &output[0], samples, f);

	fft fh(d, f, bounds(samples), fft::type::c2c);
	fft_forward(m2, m1, fh, f);

	copy(&output[0], m2, samples, f);
	wait_for(f);
	BOOST_CHECK(std::equal(output.begin(), 
				output.end(), spectrum));
	device_free(m1);
	device_free(m2);
	fft_terminate();
}

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
	fft_forward(m2, m1, fh, f);

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

