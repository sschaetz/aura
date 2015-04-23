#define BOOST_TEST_MODULE backend.fft

#include <complex>
#include <type_traits>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/config.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/fft.hpp>

#include "fft_data.hpp"

using namespace boost::aura;
using namespace boost::aura::backend;

// this is just a rudimentary set of tests, checking c2c 1d 2d and 3d on
// small data sizes witch batch=1 and batch=16

// _____________________________________________________________________________
#if 0
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
		int N = product(b);
		std::vector<cfloat> o(product(b), cfloat(0., 0.));

		device_array<cfloat> id(b, d);
		device_array<cfloat> od(b, d);

		fft fh(d, f, b, fft::type::c2c);
		copy(id.begin(), &signal_1d_4[0], product(b), f);
		fft_forward(id, od, fh, f);
		copy(&o[0], od.begin(), product(b), f);
		wait_for(f);
		BOOST_CHECK(std::equal(o.begin(), o.end(), spectrum_1d_4));
	
		copy(id.begin(), &spectrum_1d_4[0], product(b), f);
		fft_inverse(id, od, fh, f);
		copy(&o[0], od.begin(), product(b), f);
		wait_for(f);

		std::vector<cfloat> scaled(signal_1d_4, signal_1d_4+N);
#ifdef AURA_BACKEND_CUDA
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
#endif
		BOOST_CHECK(std::equal(o.begin(), o.end(), scaled.begin()));


	}

	// 2d
	{
		std::size_t dim = 
			std::sqrt(std::extent<decltype(signal_2d_4)>::value);
		bounds b(dim, dim);
		int N = product(b);
		std::vector<cfloat> o(product(b), cfloat(0., 0.));

		device_array<cfloat> id(b, d);
		device_array<cfloat> od(b, d);

		fft fh(d, f, b, fft::type::c2c);
		copy(id.begin(), &signal_2d_4[0], product(b), f);
		fft_forward(id, od, fh, f);
		copy(&o[0], od.begin(), product(b), f);
		wait_for(f);
		BOOST_CHECK(std::equal(o.begin(), o.end(), spectrum_2d_4));

		copy(id.begin(), &spectrum_2d_4[0], product(b), f);
		fft_inverse(id, od, fh, f);
		copy(&o[0], od.begin(), product(b), f);
		wait_for(f);

		std::vector<cfloat> scaled(signal_2d_4, signal_2d_4+N);
#ifdef AURA_BACKEND_CUDA
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
#endif
		BOOST_CHECK(std::equal(o.begin(), o.end(), scaled.begin()));


	}
	// 3d
	{
		std::size_t dim = 
			std::cbrt(std::extent<decltype(signal_3d_4)>::value);
		bounds b(dim, dim, dim);
		int N = product(b);
		std::vector<cfloat> o(product(b), cfloat(0., 0.));

		device_array<cfloat> id(b, d);
		device_array<cfloat> od(b, d);

		fft fh(d, f, b, fft::type::c2c);
		copy(id.begin(), &signal_3d_4[0], product(b), f);
		fft_forward(id, od, fh, f);
		copy(&o[0], od.begin(), product(b), f);
		wait_for(f);
		BOOST_CHECK(std::equal(o.begin(), o.end(), spectrum_3d_4));

		copy(id.begin(), &spectrum_3d_4[0], product(b), f);
		fft_inverse(id, od, fh, f);
		copy(&o[0], od.begin(), product(b), f);
		wait_for(f);

		std::vector<cfloat> scaled(signal_3d_4, signal_3d_4+N);
#ifdef AURA_BACKEND_CUDA
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
#endif
		BOOST_CHECK(std::equal(o.begin(), o.end(), scaled.begin()));

	}
	fft_terminate();
}

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(batched) 
{
	initialize();
	fft_initialize(); 
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);

	device d(0);
	feed f(d); 
	
	// batchsize
	int bs = 16;

	// 1d
	{
		std::size_t dim = std::extent<decltype(signal_1d_4)>::value;
		bounds b(dim);
		int N = product(b);
		std::vector<cfloat> o(product(b)*bs, cfloat(0., 0.));

		device_array<cfloat> id(bounds(b, bs), d);
		device_array<cfloat> od(bounds(b, bs), d);

		fft fh(d, f, bounds(b), fft::type::c2c, bs);
		for (int i=0; i<bs; i++) {
			copy(id.begin()+i*product(b), &signal_1d_4[0],
					product(b), f);
		}
		fft_forward(id, od, fh, f);
		copy(&o[0], od.begin(), product(b)*bs, f);
		wait_for(f);
		
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*dim,
						o.begin()+(i+1)*dim,
						spectrum_1d_4));
		}

		for (int i=0; i<bs; i++) {
			copy(id.begin()+i*product(b), &spectrum_1d_4[0],
					product(b), f);
		}
		fft_inverse(id, od, fh, f);
		copy(&o[0], od.begin(), product(b)*bs, f);
		wait_for(f);
		
		std::vector<cfloat> scaled(signal_1d_4, signal_1d_4+N);
#ifdef AURA_BACKEND_CUDA
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
#endif	
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
		std::vector<cfloat> o(product(b)*bs, cfloat(0., 0.));

		device_array<cfloat> id(bounds(b, bs), d);
		device_array<cfloat> od(bounds(b, bs), d);

		fft fh(d, f, bounds(b), fft::type::c2c, bs);
		for (int i=0; i<bs; i++) {
			copy(id.begin()+i*product(b), &signal_2d_4[0],
					product(b), f);
		}
		fft_forward(id, od, fh, f);
		copy(&o[0], od.begin(), product(b)*bs, f);
		wait_for(f);
		
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*product(b),
						o.begin()+(i+1)*product(b),
						spectrum_2d_4));
		}

		for (int i=0; i<bs; i++) {
			copy(id.begin()+i*product(b), &spectrum_2d_4[0],
					product(b), f);
		}
		fft_inverse(id, od, fh, f);
		copy(&o[0], od.begin(), product(b)*bs, f);
		wait_for(f);
		
		std::vector<cfloat> scaled(signal_2d_4, signal_2d_4+N);
#ifdef AURA_BACKEND_CUDA
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
#endif	
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
		std::vector<cfloat> o(product(b)*bs, cfloat(0., 0.));

		device_array<cfloat> id(bounds(b, bs), d);
		device_array<cfloat> od(bounds(b, bs), d);

		fft fh(d, f, bounds(b), fft::type::c2c, bs);
		for (int i=0; i<bs; i++) {
			copy(id.begin()+i*product(b), &signal_3d_4[0],
					product(b), f);
		}
		fft_forward(id, od, fh, f);
		copy(&o[0], od.begin(), product(b)*bs, f);
		wait_for(f);
		
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*product(b),
						o.begin()+(i+1)*product(b),
						spectrum_3d_4));
		}


		for (int i=0; i<bs; i++) {
			copy(id.begin()+i*product(b), &spectrum_3d_4[0],
					product(b), f);
		}
		fft_inverse(id, od, fh, f);
		copy(&o[0], od.begin(), product(b)*bs, f);
		wait_for(f);
		
		std::vector<cfloat> scaled(signal_3d_4, signal_3d_4+N);
#ifdef AURA_BACKEND_CUDA
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
#endif	
		for(int i=0; i<bs; i++) {
			BOOST_CHECK(std::equal(o.begin()+i*N,
						o.begin()+(i+1)*N,
						scaled.begin()));

		}
	}
	
	fft_terminate();
}
#endif
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(range) 
{
	initialize();
	fft_initialize(); 
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);

	device d(0);
	feed f(d); 

	// 1d
	{
		bounds b(std::extent<decltype(signal_1d_4)>::value, 10);
		bounds bsub(std::extent<decltype(signal_1d_4)>::value);

		int N = product(b);
		std::vector<cfloat> o(N, cfloat(0., 0.));

		device_array<cfloat> id(b, d);
		device_array<cfloat> od(b, d);

		// ranges
		auto idr = id(slice(_, 4));
		auto odr = od(slice(_, 4));

		fft fh(d, f, bsub, fft::type::c2c);
		copy(idr.begin(), &signal_1d_4[0], product(bsub), f);
		fft_forward(idr, odr, fh, f);
		copy(&o[0], od.begin(), product(b), f);
		wait_for(f);
		for (auto x : o) {
			std::cout << x << std::endl;
		}
		/*
		BOOST_CHECK(std::equal(o.begin(), o.end(), spectrum_1d_4));
	
		copy(id.begin(), &spectrum_1d_4[0], product(b), f);
		fft_inverse(id, od, fh, f);
		copy(&o[0], od.begin(), product(b), f);
		wait_for(f);

		std::vector<cfloat> scaled(signal_1d_4, signal_1d_4+N);
#ifdef AURA_BACKEND_CUDA
		std::transform(scaled.begin(), scaled.end(), scaled.begin(),
				[&](const cfloat& a) {
					return cfloat(a.real()*(float)N, 
						a.imag()*(float)N);
				});
#endif
		BOOST_CHECK(std::equal(o.begin(), o.end(), scaled.begin()));
		*/

	}

	fft_terminate();
}

