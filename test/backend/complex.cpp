
#define BOOST_TEST_MODULE backend.kernel

#include <cstring>
#include <complex>
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
	std::size_t xdim = 4;
	std::size_t ydim = 4;

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
		cfloat c ((float)i, (float)i*0.1);
		cfloat c1 = conj(cfloat(std::imag(c), std::real(c))) + c;
		cfloat c2 = conj(cfloat(std::imag(c), std::real(c))) - c;
		cfloat c3 = (c1*c2)/c;	
		a1[i] = std::abs(c3);
		std::cout << a2[i] << " " << a1[i] << std::endl;
	}
	BOOST_CHECK(std::equal(a1.begin(), a1.end(), a2.begin()));
	device_free(mem);
}

