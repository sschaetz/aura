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
	BOOST_CHECK_ASSERT(0 < num);
	device d(0);  
	feed f(d);
	device_ptr<float> ptr1 = device_malloc<float>(15, d);
	device_ptr<float> ptr2 = ptr1;
	BOOST_CHECK(ptr1 == ptr2);
	ptr1 = ptr1+1;
	BOOST_CHECK(ptr1 != ptr2);
	++ptr2;
	BOOST_CHECK(ptr1 == ptr2);
	device_free(ptr1);
	BOOST_CHECK(ptr1 == nullptr);
	BOOST_CHECK(nullptr == ptr1);
	BOOST_CHECK(ptr2 != nullptr);
	BOOST_CHECK(nullptr != ptr2);

	device_ptr<float> ptr3(ptr2);
	BOOST_CHECK(ptr3 == ptr2);
}

// allocation_free 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(allocation_free) 
{
	initialize();
	int num = device_get_count();
	BOOST_CHECK_ASSERT(0 < num);
	device d(0);  
	feed f(d);
	device_ptr<float> ptr1= device_malloc<float>(16, d);
	device_ptr<float> ptr2;
	BOOST_CHECK(ptr1 != ptr2);
	device_free<float>(ptr1);
	BOOST_CHECK(ptr1 == ptr2);
}
