#define BOOST_TEST_MODULE misc.profile

#include <unistd.h>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <aura/misc/sequence.hpp>

using namespace aura;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
  sequence<std::size_t, 3> s("10:+3:17,100:*2:2000");
  // should generate (10, 100) (13, 200) (16, 400) (false)
  bool good;
  svec<std::size_t, 3> dims;
  
  std::tie(dims, good) = s.next();
  BOOST_CHECK(dims[0] == 10);
  BOOST_CHECK(dims[1] == 100);
  BOOST_CHECK(good == true);
  
  std::tie(dims, good ) = s.next();
  BOOST_CHECK(dims[0] == 13);
  BOOST_CHECK(dims[1] == 200);
  BOOST_CHECK(good == true );

  std::tie(dims, good) = s.next();
  BOOST_CHECK(dims[0] == 16);
  BOOST_CHECK(dims[1] == 400);
  BOOST_CHECK(good == true);
  
  std::tie(dims, good) = s.next();
  BOOST_CHECK(good == false);
}

