#define BOOST_TEST_MODULE backend.host_allocator

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/host_allocator.hpp>

using namespace boost::aura::backend;

// basic 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d0(0);
	feed f0(d0);
	boost::aura::host_allocator<float> a0(f0);
	
	float* ptr0 = a0.allocate(100);
	memory m0 = a0.unmap(ptr0);
	float* ptr1 = a0.map(m0);
	BOOST_CHECK(ptr0 == ptr1);
	memory m1 = a0.unmap(ptr1);
	BOOST_CHECK(m0 == m1);
	a0.deallocate(ptr0, 100);
}

// stdvector 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(stdvector) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d0(0);
	feed f0(d0);
	boost::aura::host_allocator<float> a0(f0);
	std::vector<float, boost::aura::host_allocator<float>> v0(100, a0);
	memory m0 = v0.get_allocator().unmap(&v0[0]);
	float* ptr0 = v0.get_allocator().map(m0);
	BOOST_CHECK(ptr0 == &v0[0]);
}

// nasty 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(nasty) 
{
	initialize();
	int num = device_get_count();
	BOOST_REQUIRE(num > 0);
	device d0(0);
	feed f0(d0);
	
	boost::aura::host_allocator<float> a0(f0);
	boost::aura::host_allocator<float> a1(f0);
	std::vector<float, boost::aura::host_allocator<float>> v0(100, a0);
	std::vector<float, boost::aura::host_allocator<float>> v1(100, a1);
	
	float* p0 = &v0[0];
	float* p1 = &v1[0];
	
	std::swap(v0, v1);
	
	memory m0 = v0.get_allocator().unmap(p1);
	memory m1 = v1.get_allocator().unmap(p0);
	
	float* p2 = v0.get_allocator().map(m0);
	float* p3 = v1.get_allocator().map(m1);

	BOOST_CHECK(p0 == p3);
	BOOST_CHECK(p1 == p2);
	BOOST_CHECK(&v0[0] == p2);
	BOOST_CHECK(&v1[0] == p3);
}

