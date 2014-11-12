#define BOOST_TEST_MODULE backend.basic 

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>

using namespace aura::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d(0);  
	feed f(d);
}

// memorypingpong 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(memorypingpong) 
{
	initialize();

	int num = device_get_count();
	BOOST_REQUIRE(0 < num);

	device d(0);  
	feed f(d);
	int testsize = 512; 
	std::vector<float> a1(testsize, 42.);
	std::vector<float> a2(testsize);
	device_ptr<float> m = device_malloc<float>(testsize, d);
	copy(m, &a1[0], testsize, f); 
	copy(&a2[0], m, testsize, f);
	wait_for(f);
	BOOST_CHECK(std::equal(a1.begin(), a1.end(), a2.begin()));
	device_free(m);
}

