#define BOOST_TEST_MODULE detail.svec

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp> 
#include <aura/detail/svec.hpp>

using namespace aura;

typedef svec<int, 3> dim3;

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
  // this should assert:
  //dim3 d4(0,1,2,3);
}


