#define BOOST_TEST_MODULE backend.kernel

#include <cstring>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/config.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;
using namespace boost::aura::backend;

const char * kernel_file = AURA_UNIT_TEST_LOCATION"kernel.cc";

// basic
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

// basic2
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic2) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	device d(0); 
	kernel k = d.load_from_file("noarg", kernel_file,
			AURA_BACKEND_COMPILE_FLAGS);
	(void)k; 
}


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

// invoke_simple2
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(invoke_simple2) 
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

	kernel k = d.load_from_file("simple_add", kernel_file, 
			AURA_BACKEND_COMPILE_FLAGS); 
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

// invoke_shared
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(invoke_shared) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	device d(0);  
	feed f(d);
	std::size_t xdim = 8;
	std::size_t ydim = 8;
	std::size_t b = 4;

	std::vector<float> a1(xdim*ydim, 0.);
	std::vector<float> a2(xdim*ydim);

	module mod = create_module_from_file(kernel_file, d, 
	AURA_BACKEND_COMPILE_FLAGS);
	print_module_build_log(mod, d);
	kernel k = create_kernel(mod, "simple_shared"); 
	device_ptr<float> mem = device_malloc<float>(xdim*ydim, d);

	copy(mem, &a1[0], xdim*ydim, f); 
	invoke(k, mesh(ydim, xdim), bundle(b), args(mem.get()), f);
	copy(&a2[0], mem, xdim*ydim, f);
	wait_for(f);

	// generate correct data
	int cur = 0;
	std::generate(a1.begin(), a1.end(), [&]() { return (float)cur++; } );
	for (std::size_t i=0; i<a1.size(); i+=b) {
		std::reverse(a1.begin()+i, a1.begin()+i+b);	
	}
	BOOST_CHECK(std::equal(a1.begin(), a1.end(), a2.begin()));
	device_free(mem);
}

// invoke_atomic
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(invoke_atomic) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);
	device d(0);  
	feed f(d);
	std::size_t xdim = 8;
	std::size_t ydim = 8;
	std::size_t b = 4;

	std::vector<float> a1(xdim*ydim, 0.);
	std::vector<float> a2(xdim*ydim);

	module mod = create_module_from_file(kernel_file, d, 
	AURA_BACKEND_COMPILE_FLAGS);
	print_module_build_log(mod, d);
	kernel k = create_kernel(mod, "simple_atomic"); 
	device_ptr<float> mem = device_malloc<float>(xdim*ydim, d);

	copy(mem, &a1[0], xdim*ydim, f); 
	invoke(k, mesh(ydim, xdim), bundle(b), args(mem.get()), f);
	copy(&a2[0], mem, xdim*ydim, f);
	wait_for(f);
	a1[0] = ydim*xdim;
	BOOST_CHECK(std::equal(a1.begin(), a1.end(), a2.begin()));
	for (auto x:a1) {
		std::cout << x << " ";
	}
	device_free(mem);
}

