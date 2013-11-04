#define BOOST_TEST_MODULE detail.svec

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp> 
#include <aura/detail/svec.hpp>

using namespace aura;

typedef svec<int, 3> dim3;

struct dummy1 {
  dummy1() {}
  dummy1(int n) {}
  dummy1 operator *=(dummy1 b) {
    return b;
  }
};

struct dummy2 {
  dummy2() {}
  dummy2(int n) {}
};

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
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

BOOST_AUTO_TEST_CASE(push_back) {
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

