#include <complex>
#include <numeric>
#include <vector>
#include <algorithm>
#include <future>

#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>

#include <boost/aura/misc/benchmark.hpp>
#include <boost/aura/misc/profile.hpp>
#include <boost/aura/misc/profile_svg.hpp>

#include <fftw3.h>

#include <boost/aura/config.hpp>
#include <boost/aura/misc/coo.hpp>

using namespace boost::aura;

typedef std::complex<float> cfloat;

boost::aura::profile::memory_sink ms;

void with_future()
{
	// load data
	auto ftr0 = std::async(std::launch::async, []() {
			boost::aura::profile::scope<boost::aura::profile::memory_sink> s(ms, "coo_read");
			return coo_read<cfloat>("input.coo");
			}
		);
	auto sftr0 = ftr0.share();
	
	// calc norm, scale
	auto ftr1 = 
		std::async(std::launch::async, [&]() {
			boost::aura::profile::scope<boost::aura::profile::memory_sink> s(ms, "norm scale");
			std::vector<cfloat> data(
				std::move(std::get<0>(sftr0.get())));
			double norm = std::sqrt(
				std::accumulate(data.begin(), data.end(), 0.0,
					[](double res, cfloat c) {
						return res + std::pow(
							(double)std::abs(c), 2);
					}
				));

			// output norm
			// output norm asynchronously
			auto ftr2 = std::async(std::launch::async, [=]() {
				boost::aura::profile::scope<boost::aura::profile::memory_sink> s(ms, "output");
					//std::cout << norm << " ";
				});

			// scale data
			std::transform(data.begin(), data.end(), data.begin(),
					[=](cfloat c) {
						return c*= norm;
					}
				);
			return make_tuple(data, std::move(ftr2));
		});
	auto sftr1 = ftr1.share();

	// generate pattern
	auto ftr3 = std::async(std::launch::async, [&]() {
			boost::aura::profile::scope<boost::aura::profile::memory_sink> s(ms, "pattern");
			std::vector<cfloat> pattern;
			auto dims = std::get<1>(sftr0.get()); 
			pattern.resize(product(dims));
			for (int y=0; y<dims[0]; y++) {
				for (int x=0; x<dims[1]; x++) {
					pattern[y*dims[1] + x] = 
						cfloat((0 == (x + y) % 2) ? 
								-1. : 1., 0.);
				}
			}

			return pattern;
		});
	
	auto ftr4 = std::async(std::launch::async, [&]() {
			boost::aura::profile::scope<boost::aura::profile::memory_sink> s(ms, "rest");
			auto& dims = std::get<1>(sftr0.get()); 
			std::vector<std::complex<float>> data(
				std::move(std::get<0>(sftr1.get())));

			// create FFT plan
			fftwf_plan fftplan = fftwf_plan_dft_2d(dims[0], dims[1],
					nullptr, nullptr, FFTW_FORWARD, 
					FFTW_ESTIMATE);
			auto pattern(std::move(ftr3.get()));
			// fft shift
			std::transform(data.begin(), data.end(), 
				pattern.begin(), data.begin(),
				std::multiplies<std::complex<float>>());

			// do FFT
			fftwf_execute_dft(fftplan, reinterpret_cast<fftwf_complex*>(&data[0]), 
					reinterpret_cast<fftwf_complex*>(&data[0]));
			
			// destroy FFT plan
			fftwf_destroy_plan(fftplan);	
			
			// fft shift
			std::transform(data.begin(), data.end(), 
				pattern.begin(), data.begin(),
				std::multiplies<std::complex<float>>());
			
			// write data to file
			coo_write(data.begin(), dims, "w_future_out.coo");
		});
	ftr4.wait();
}

#if 0
void with_future_boost()
{
	boost::basic_thread_pool pool;
	// load data
	auto ftr0 = boost::async(boost::launch::async, []() {
			return coo_read<cfloat>("input.coo");
			}
		);
	auto sftr0 = ftr0.share();
	
	// calc norm, scale
	auto ftr1 = 
		boost::async(boost::launch::async, [&]() {
			std::vector<cfloat> data(
				std::move(std::get<0>(sftr0.get())));
			double norm = std::sqrt(
				std::accumulate(data.begin(), data.end(), 0.0,
					[](double res, cfloat c) {
						return res + std::pow(
							(double)std::abs(c), 2);
					}
				));

			// output norm
			// output norm asynchronously
			auto ftr2 = boost::async(boost::launch::async, [=]() {
					//std::cout << norm << " ";
				});

			// scale data
			std::transform(data.begin(), data.end(), data.begin(),
					[=](cfloat c) {
						return c*= norm;
					}
				);
			return make_tuple(data, std::move(ftr2));
		});
	auto sftr1 = ftr1.share();

	// generate pattern
	auto ftr3 = boost::async(boost::launch::async, [&]() {
			std::vector<cfloat> pattern;
			auto dims = std::get<1>(sftr0.get()); 
			pattern.resize(product(dims));
			for (int y=0; y<dims[0]; y++) {
				for (int x=0; x<dims[1]; x++) {
					pattern[y*dims[1] + x] = 
						cfloat((0 == (x + y) % 2) ? 
								-1. : 1., 0.);
				}
			}

			return pattern;
		});
	
	auto ftr4 = boost::async(boost::launch::async, [&]() {
			auto& dims = std::get<1>(sftr0.get()); 
			std::vector<std::complex<float>> data(
				std::move(std::get<0>(sftr1.get())));

			// create FFT plan
			fftwf_plan fftplan = fftwf_plan_dft_2d(dims[0], dims[1],
					nullptr, nullptr, FFTW_FORWARD, 
					FFTW_ESTIMATE);
			auto pattern(std::move(ftr3.get()));
			// fft shift
			std::transform(data.begin(), data.end(), 
				pattern.begin(), data.begin(),
				std::multiplies<std::complex<float>>());

			// do FFT
			fftwf_execute_dft(fftplan, reinterpret_cast<fftwf_complex*>(&data[0]), 
					reinterpret_cast<fftwf_complex*>(&data[0]));
			
			// destroy FFT plan
			fftwf_destroy_plan(fftplan);	
			
			// fft shift
			std::transform(data.begin(), data.end(), 
				pattern.begin(), data.begin(),
				std::multiplies<std::complex<float>>());
			
			// write data to file
			coo_write(data.begin(), dims, "w_future_boost_out.coo");
		});
	ftr4.wait();
}
#endif


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
	//std::cout << norm << " ";

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

int main(void) {
	boost::aura::benchmark_result br;
	without_future();
	dump_svg(ms, "wof.svg");
	with_future();
	dump_svg(ms, "wf.svg");
/*	
	AURA_BENCH(without_future(), 5000000, br);
	std::cout << "without_future " << br << std::endl;
	AURA_BENCH(with_future(), 5000000, br);
	std::cout << "with_future " << br << std::endl;
	AURA_BENCH(with_future_boost(), 5000000, br);
	std::cout << "with_future_boost " << br << std::endl;
*/
}


