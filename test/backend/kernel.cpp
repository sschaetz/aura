#define BOOST_TEST_MODULE backend.kernel

#include <cstring>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>
#include <aura/config.hpp>
#include <aura/device_array.hpp>

using namespace aura;
using namespace aura::backend;

const char * kernel_file = AURA_UNIT_TEST_LOCATION"kernel.cc";

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	device d(0); 
	module m = create_module_from_file(kernel_file, d, 
	AURA_BACKEND_COMPILE_FLAGS);
	kernel k = create_kernel(m, "noarg");
	(void)k; 
}

// basic

// invoke_simple
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(invoke_simple) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	device d(0);  
	feed f(d);
	std::size_t xdim = 16;
	std::size_t ydim = 16;

	std::vector<float> a1(xdim*ydim, 41.);
	std::vector<float> a2(xdim*ydim);

	module mod = create_module_from_file(kernel_file, d, 
	AURA_BACKEND_COMPILE_FLAGS);
	print_module_build_log(mod, d);
	kernel k = create_kernel(mod, "simple_add"); 
	device_ptr<float> mem = device_malloc<float>(xdim*ydim, d);

	copy(mem, &a1[0], xdim*ydim, f); 
	invoke(k, mesh(ydim, xdim), bundle(xdim), args(mem.get()), f);
	copy(&a2[0], mem, xdim*ydim, f);
	wait_for(f);

	for(std::size_t i=0; i<a1.size(); i++) {
		a1[i] += 1.0;
	}
	BOOST_CHECK(std::equal(a1.begin(), a1.end(), a2.begin()));
	device_free(mem);
}

// invoke_noarg
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(invoke_noarg) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	device d(0);  
	feed f(d);
	std::size_t xdim = 16;
	std::size_t ydim = 16;

	module mod = create_module_from_file(kernel_file, d, 
	AURA_BACKEND_COMPILE_FLAGS);

	kernel k = create_kernel(mod, "noarg"); 
	invoke(k, mesh(ydim), bundle(xdim), f);
	wait_for(f);
}

// invoke_nomesh
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(invoke_nomesh) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	
	device d(0);  
	feed f(d);
	bounds b(64, 64, 32);
	std::vector<float> a1(product(b), 41.);
	std::vector<float> a2(product(b));


	module mod = create_module_from_file(kernel_file, d,
		AURA_BACKEND_COMPILE_FLAGS);
	print_module_build_log(mod, d);

	kernel k = create_kernel(mod, "simple_add"); 
	device_array<float> v(b, d);

	copy(v.begin(), &a1[0], product(b), f); 
	invoke(k, b, args(v.begin().get()), f);
	copy(&a2[0], v.begin(), product(b), f);
	wait_for(f);

	for(std::size_t i=0; i<a1.size(); i++) {
		a1[i] += 1.0;
	}
	BOOST_CHECK(std::equal(a1.begin(), a1.end(), a2.begin()));
}

