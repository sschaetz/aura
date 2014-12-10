
#define BOOST_TEST_MODULE backend.kernel_complex

#include <iomanip>
#include <limits>
#include <cstring>
#include <complex>
#include <numeric>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/config.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;

const char * kernel_file = AURA_UNIT_TEST_LOCATION"complex.cc";

// complex_arithmetic
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(complex_single) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	device d(0);  
	feed f(d);
	std::size_t xdim = 64;
	std::size_t ydim = 64;

	std::vector<float> a1(xdim*ydim, 41.);
	std::vector<float> a2(xdim*ydim);

	module mod = create_module_from_file(kernel_file, d, 
	AURA_BACKEND_COMPILE_FLAGS);
	print_module_build_log(mod, d);
	kernel k = create_kernel(mod, "complex_single"); 
	device_ptr<float> mem = device_malloc<float>(xdim*ydim, d);

	copy(mem, &a1[0], xdim*ydim, f); 
	invoke(k, mesh(ydim, xdim), bundle(xdim), args(mem.get()), f);
	copy(&a2[0], mem, xdim*ydim, f);
	wait_for(f);
	typedef std::complex<float> cfloat;

	for(std::size_t i=0; i<a1.size(); i++) {
		cfloat c ((float)i+1, (float)(i+1)*0.1);
		cfloat c1 = conj(cfloat(std::imag(c), std::real(c))) + c;
		cfloat c2 = conj(cfloat(std::imag(c), std::real(c))) - c;
		cfloat c3 = (c1*c2)/c;	
		a1[i] = std::abs(c3);
	}
	float hsum = std::abs(std::accumulate(a1.begin(), a1.end(), .0));
	float dsum = std::abs(std::accumulate(a2.begin(), a2.end(), .0));
	float err = std::abs(hsum - dsum);
	BOOST_CHECK(err < std::numeric_limits<float>::epsilon());
	device_free(mem);
}

// TODO: test double precision
// TODO: do more comprehensive tests

