#define BOOST_TEST_MODULE detail.svec

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp> 
#include <boost/aura/detail/svec.hpp>

using namespace boost::aura;

typedef svec<int, 3> dim3;

struct dummy1 
{
	dummy1() {}
	dummy1(int n) {}
	dummy1 operator *=(dummy1 b) 
	{
		return b;
	}
};

struct dummy2 
{
	dummy2() {}
	dummy2(int n) {}
};

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	dim3 d3(0,1,2);
	BOOST_CHECK(d3.size()==3);
	for(int i=0; i<(int)d3.size(); i++) {
		BOOST_CHECK(d3[i]==i);
		d3[i] = -i;
		BOOST_CHECK(d3[i]==-i);
	}
	dim3 d2(0,1);
	BOOST_CHECK(d2.size() == 2);

	dim3 dp(4,4,4);
	BOOST_CHECK(product(dp)  == 4*4*4);

	svec<dummy1, 3> sd(dummy1(12), dummy1(13), dummy1(14));
	dummy1 foo = product(sd);
	(void)foo; 

	// this should assert:
	//dim3 d4(0,1,2,3);

	svec<dummy2, 3> sdd(dummy2(12), dummy2(13), dummy2(14));
	// this should assert:
	//dummy2 fooo = product(sdd);
}

// push_back 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(push_back) 
{
	dim3 d3;
	BOOST_CHECK(d3.size()==0);
	d3.push_back(42);
	BOOST_CHECK(d3.size()==1);
	BOOST_CHECK(d3[0]== 42);
	d3.push_back(43);
	BOOST_CHECK(d3.size()==2);
	BOOST_CHECK(d3[1]== 43);
	d3.push_back(44);
	BOOST_CHECK(d3.size()==3);
	BOOST_CHECK(d3[2]== 44);
	// this should assert:
	//d3.push_back(45);
}

// take_test 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(take_test) 
{
	svec<int, 16> a(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

	auto b = take(2, a);
	svec<int, 16> a_take_2(0, 1);
	BOOST_CHECK(b == a_take_2);
	
	b = take(1, a);
	svec<int, 16> a_take_1(0);
	BOOST_CHECK(b == a_take_1);
	
	b = take(0, a);
	svec<int, 16> a_take_0;
	BOOST_CHECK(b == a_take_0);

	b = take(16, a);
	BOOST_CHECK(b == a);

	b = take(17, a);
	BOOST_CHECK(b == a);
}

// drop_test 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(drop_test) 
{
	svec<int, 16> a(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

	auto b = drop(2, a);
	svec<int, 16> a_drop_2(14, 15);
	BOOST_CHECK(b == a_drop_2);
	
	b = drop(1, a);
	svec<int, 16> a_drop_1(15);
	BOOST_CHECK(b == a_drop_1);
	
	b = drop(0, a);
	svec<int, 16> a_drop_0(a);
	BOOST_CHECK(b == a_drop_0);

	b = drop(17, a);
	BOOST_CHECK(b == a);

	b = drop(20, a);
	BOOST_CHECK(b == a);

}

// split_at_test 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(split_at_test) 
{
	svec<int, 16> a(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	svec<int, 16> b;
	svec<int, 16> c;

	std::tie(b, c) = split_at(6, a);
	svec<int, 16> b_split_6(0, 1, 2, 3, 4, 5);
	svec<int, 16> c_split_6(6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	BOOST_CHECK(b == b_split_6);
	BOOST_CHECK(c == c_split_6);

	std::tie(b, c) = split_at(0, a);
	svec<int, 16> b_split_0;
	svec<int, 16> c_split_0(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
			10, 11, 12, 13, 14, 15);
	BOOST_CHECK(b == b_split_0);
	BOOST_CHECK(c == c_split_0);

	std::tie(b, c) = split_at(16, a);
	svec<int, 16> b_split_16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
			10, 11, 12, 13, 14, 15);
	svec<int, 16> c_split_16;
	BOOST_CHECK(b == b_split_16);
	BOOST_CHECK(c == c_split_16);

	std::tie(b, c) = split_at(20, a);
	svec<int, 16> b_split_20(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
			10, 11, 12, 13, 14, 15);
	svec<int, 16> c_split_20;
	BOOST_CHECK(b == b_split_20);
	BOOST_CHECK(c == c_split_20);

}

