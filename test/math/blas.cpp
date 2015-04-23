#define BOOST_TEST_MODULE backend.blas

#include <vector>
#include <type_traits>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/config.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/blas.hpp>


using namespace boost::aura;
using namespace boost::aura::backend;

// this is just a rudimentary set of tests, checking c2c 1d 2d and 3d on
// small data sizes witch batch=1 and batch=16

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	initialize();
	blas_initialize(); 
	int num = device_get_count();
	BOOST_REQUIRE(0 < num);

	device d(0);
	feed f(d); 
	
	std::vector<float> A(product(bounds(10,10)), 42.);
	std::vector<float> x(10, 0.42);
	std::vector<float> y(10, 1.0);

	device_array<float> Ad(bounds(10,10), d);
	device_array<float> xd(10, d);
	device_array<float> yd(10, d);

	copy(A, Ad, f);
	copy(x, xd, f);
	copy(y, yd, f);
	
	gemv(Ad, xd, yd, f);

	copy(yd, y, f);
	wait_for(f);

	for (auto r : y)
	{
		std::cout << r << std::endl;
	}

	blas_terminate();
}

