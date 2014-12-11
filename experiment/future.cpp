#include <complex>
#include <numeric>
#include <vector>
#include <algorithm>

// this experiment breaks the build, it is not yet finished
#if 0
#include <fftw3.h>

#include <boost/aura/config.hpp>
#include <boost/aura/misc/coo.hpp>

using namespace boost::aura;



void with_future()
{	
	//auto in = boost::async(pool, &coo_read<std::complex<float>>, "input.coo");
	/*
	// load data
	auto in = coo_read<std::complex<float>>("input.coo");

	auto& data = std::get<0>(in);
	auto dims = std::get<1>(in);

	// calc norm
	double norm = std::sqrt(std::accumulate(data.begin(), data.end(), 0.0,
				[](double res, std::complex<float> c) {
					return res + std::pow(
						(double)std::abs(c), 2); 
				}
			)
		);
	
	// scale data
	std::transform(data.begin(), data.end(), data.begin(), 
			[=](std::complex<float> c) {
				return c*= norm;
			}
		);

	// output norm
	std::cout << norm << std::endl;

	// generate pattern
	std::vector<std::complex<float>> pattern;
	pattern.resize(product(dims));
	for (int y=0; y<dims[0]; y++) {
		for (int x=0; x<dims[1]; x++) {
			pattern[y*dims[1] + x] = 
				std::complex<float>((0 == (x + y) % 2) ? 
						-1. : 1., 0.);
		}
	}

	// multiply pattern
	std::transform(data.begin(), data.end(), pattern.begin(), data.begin(),
			std::multiplies<std::complex<float>>());

	// create FFT plan
	fftwf_plan fftplan = fftwf_plan_dft_2d(dims[0], dims[1],
			nullptr, nullptr, FFTW_FORWARD, 
			FFTW_ESTIMATE);
	// do FFT
	fftwf_execute_dft(fftplan, reinterpret_cast<fftwf_complex*>(&data[0]), 
			reinterpret_cast<fftwf_complex*>(&data[0]));
	
	// destroy FFT plan
	fftwf_destroy_plan(fftplan);	

	// multiply pattern
	std::transform(data.begin(), data.end(), pattern.begin(), data.begin(),
			std::multiplies<std::complex<float>>());
	
	// write data to file
	coo_write(data.begin(), dims, "wo_future_out.coo");
	*/
}


void without_future()
{	
	// load data
	auto in = coo_read<std::complex<float>>("input.coo");

	auto& data = std::get<0>(in);
	auto dims = std::get<1>(in);

	// calc norm
	double norm = std::sqrt(std::accumulate(data.begin(), data.end(), 0.0,
				[](double res, std::complex<float> c) {
					return res + std::pow(
						(double)std::abs(c), 2); 
				}
			)
		);
	
	// scale data
	std::transform(data.begin(), data.end(), data.begin(), 
			[=](std::complex<float> c) {
				return c*= norm;
			}
		);

	// output norm
	std::cout << norm << std::endl;

	// generate pattern
	std::vector<std::complex<float>> pattern;
	pattern.resize(product(dims));
	for (int y=0; y<dims[0]; y++) {
		for (int x=0; x<dims[1]; x++) {
			pattern[y*dims[1] + x] = 
				std::complex<float>((0 == (x + y) % 2) ? 
						-1. : 1., 0.);
		}
	}

	// multiply pattern
	std::transform(data.begin(), data.end(), pattern.begin(), data.begin(),
			std::multiplies<std::complex<float>>());

	// create FFT plan
	fftwf_plan fftplan = fftwf_plan_dft_2d(dims[0], dims[1],
			nullptr, nullptr, FFTW_FORWARD, 
			FFTW_ESTIMATE);
	// do FFT
	fftwf_execute_dft(fftplan, reinterpret_cast<fftwf_complex*>(&data[0]), 
			reinterpret_cast<fftwf_complex*>(&data[0]));
	
	// destroy FFT plan
	fftwf_destroy_plan(fftplan);	

	// multiply pattern
	std::transform(data.begin(), data.end(), pattern.begin(), data.begin(),
			std::multiplies<std::complex<float>>());
	
	// write data to file
	coo_write(data.begin(), dims, "wo_future_out.coo");
}

#endif

int main(void) {
#if 0
	without_future();
	with_future();
#endif
}


