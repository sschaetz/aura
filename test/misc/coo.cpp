
#define BOOST_TEST_MODULE misc.profile 

#include <algorithm>
#include <vector>
#include <complex>
#include <boost/test/unit_test.hpp>
#include <aura/bounds.hpp>
#include <aura/misc/coo.hpp>


// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	aura::bounds b0(16, 16, 1, 1, 100);
	std::vector<std::complex<float>> d0(aura::product(b0), 
			std::complex<float>(42.0, 21.0));

	aura::coo_write(d0.begin(), b0, "test.coo");

	aura::bounds b1;
	std::vector<std::complex<float>> d1;
	std::tie(d1, b1) = aura::coo_read<std::complex<float>>("test.coo");

	BOOST_CHECK(std::equal(d1.begin(), d1.end(), d0.begin()));
	BOOST_CHECK(b0 == b1);
}

