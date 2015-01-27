#define BOOST_TEST_MODULE backend.fft_host

#include <vector>
#include <complex>
#include <type_traits>
#include <boost/test/unit_test.hpp>
#include <boost/aura/ext/fftw.hpp>
#include <boost/aura/backend.hpp>

#include "fft_data.hpp"

using namespace boost::aura;

// this is just a rudimentary set of tests, checking c2c 1d 2d and 3d on
// small data sizes witch batch=1 and batch=16

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	fftw::fft_initialize(); 

	// 1d
	{
		bounds b(std::extent<decltype(signal_1d_4)>::value);
		int N = product(b);
		
		std::vector<cfloat> o(N, cfloat(0., 0.));

		fftw::fft fh(b, fftw::fft::type::c2c);
		fftw::fft_forward(signal_1d_4, o.begin(), fh);
		BOOST_CHECK(std::equal(o.begin(), o.end(), spectrum_1d_4));

		fftw::fft_inverse(spectrum_1d_4, o.begin(), fh);
		std::vector<cfloat> scaled(signal_1d_4, signal_1d_4+N);
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
		BOOST_CHECK(std::equal(o.begin(), o.end(), scaled.begin()));
		
	
	}

	// 2d
	{
		std::size_t dim = 
			std::sqrt(std::extent<decltype(signal_2d_4)>::value);
		bounds b(dim, dim);
		int N = product(b);
		std::vector<cfloat> o(N, cfloat(0., 0.));

		fftw::fft fh(b, fftw::fft::type::c2c);
		fft_forward(signal_2d_4, o.begin(), fh);
		BOOST_CHECK(std::equal(o.begin(), o.end(), spectrum_2d_4));

		fftw::fft_inverse(spectrum_2d_4, o.begin(), fh);
		std::vector<cfloat> scaled(signal_2d_4, signal_2d_4+N);
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
		BOOST_CHECK(std::equal(o.begin(), o.end(), scaled.begin()));

	}
	// 3d
	{
		std::size_t dim = 
			std::cbrt(std::extent<decltype(signal_3d_4)>::value);
		bounds b(dim, dim, dim);
		int N = product(b);
		std::vector<cfloat> o(N, cfloat(0., 0.));

		fftw::fft fh(b, fftw::fft::type::c2c);
		fftw::fft_forward(signal_3d_4, o.begin(), fh);
		BOOST_CHECK(std::equal(o.begin(), o.end(), spectrum_3d_4));

		fftw::fft_inverse(spectrum_3d_4, o.begin(), fh);
		std::vector<cfloat> scaled(signal_3d_4, signal_3d_4+N);
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
		BOOST_CHECK(std::equal(o.begin(), o.end(), scaled.begin()));

	}
	fftw::fft_terminate();

}

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(batched) 
{
	fftw::fft_initialize(); 
	
	// batchsize
	int bs = 16;

	// 1d
	{
		std::size_t dim = std::extent<decltype(signal_1d_4)>::value;
		bounds b(dim);
		int N = product(b);
		std::vector<cfloat> in(N*bs, cfloat(0., 0.));
		std::vector<cfloat> o(N*bs, cfloat(0., 0.));

		fftw::fft fh(bounds(b), fftw::fft::type::c2c, bs);
		for (int i=0; i<bs; i++) {
			std::copy(signal_1d_4, signal_1d_4 + N,
					&in[i*N]);
		}
		fftw::fft_forward(in.begin(), o.begin(), fh);
		
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*dim,
						o.begin()+(i+1)*dim,
						spectrum_1d_4));
		}
		
		for (int i=0; i<bs; i++) {
			std::copy(spectrum_1d_4, spectrum_1d_4 + N,
					&in[i*N]);
		}
		fft_inverse(in.begin(), o.begin(), fh);
		
		std::vector<cfloat> scaled(signal_1d_4, signal_1d_4+N);
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
	
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*N,
						o.begin()+(i+1)*N,
						scaled.begin()));
		}
	}

	// 2d
	{
		std::size_t dim = 
			std::sqrt(std::extent<decltype(signal_2d_4)>::value);
		bounds b(dim, dim);

		int N = product(b);
		std::vector<cfloat> in(N*bs, cfloat(0., 0.));
		std::vector<cfloat> o(N*bs, cfloat(0., 0.));

		fftw::fft fh(bounds(b), fftw::fft::type::c2c, bs);
		for (int i=0; i<bs; i++) {
			std::copy(signal_2d_4, signal_2d_4 + N,
					&in[i*N]);
		}
		fftw::fft_forward(in.begin(), o.begin(), fh);
		
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*N,
						o.begin()+(i+1)*N,
						spectrum_2d_4));
		}
		
		for (int i=0; i<bs; i++) {
			std::copy(spectrum_2d_4, spectrum_2d_4 + N,
					&in[i*N]);
		}
		fft_inverse(in.begin(), o.begin(), fh);
		
		std::vector<cfloat> scaled(signal_2d_4, signal_2d_4+N);
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
	
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*N,
						o.begin()+(i+1)*N,
						scaled.begin()));
		}
	}

	// 3d
	{
		std::size_t dim = 
			std::cbrt(std::extent<decltype(signal_3d_4)>::value);
		bounds b(dim, dim, dim);

		int N = product(b);
		std::vector<cfloat> in(N*bs, cfloat(0., 0.));
		std::vector<cfloat> o(N*bs, cfloat(0., 0.));

		fftw::fft fh(bounds(b), fftw::fft::type::c2c, bs);
		for (int i=0; i<bs; i++) {
			std::copy(signal_3d_4, signal_3d_4 + N,
					&in[i*N]);
		}
		fftw::fft_forward(in.begin(), o.begin(), fh);
		
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*N,
						o.begin()+(i+1)*N,
						spectrum_3d_4));
		}
		
		for (int i=0; i<bs; i++) {
			std::copy(spectrum_3d_4, spectrum_3d_4 + N,
					&in[i*N]);
		}
		fft_inverse(in.begin(), o.begin(), fh);
		
		std::vector<cfloat> scaled(signal_3d_4, signal_3d_4+N);
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
	
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*N,
						o.begin()+(i+1)*N,
						scaled.begin()));
		}
	}


}

