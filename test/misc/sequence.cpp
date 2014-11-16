#define BOOST_TEST_MODULE misc.profile

#include <unistd.h>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <boost/aura/misc/sequence.hpp>

using namespace boost::aura;

// basic
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(basic) {
  sequence<std::size_t, 3> s("10:+3:17;100:*2:2000");
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


// generator 
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(generator) {
  std::vector<boost::aura::svec<std::size_t, 3> > vec = 
    generate_sequence<std::size_t, 3>("10:+3:17;100:*2:2000");
  // should generate (10, 100) (13, 200) (16, 400) 
  BOOST_CHECK(vec.size() == 3); 
  BOOST_CHECK(vec[0][0] == 10);
  BOOST_CHECK(vec[0][1] == 100);
  BOOST_CHECK(vec[1][0] == 13);
  BOOST_CHECK(vec[1][1] == 200);
  BOOST_CHECK(vec[2][0] == 16);
  BOOST_CHECK(vec[2][1] == 400);
}

// explicitvals
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(explicitvals) {
  std::vector<boost::aura::svec<std::size_t, 3> > vec = 
    generate_sequence<std::size_t, 3>("(100,200,300);4:+1:8");
  // should generate (100, 4) (200, 5) (300, 6)
  BOOST_CHECK(vec.size() == 3); 
  BOOST_CHECK(vec[0][0] == 100);
  BOOST_CHECK(vec[0][1] == 4);
  BOOST_CHECK(vec[1][0] == 200);
  BOOST_CHECK(vec[1][1] == 5);
  BOOST_CHECK(vec[2][0] == 300);
  BOOST_CHECK(vec[2][1] == 6);
  
  vec = generate_sequence<std::size_t, 3>("(100,200,300);(1,2,3)");
  // should generate (100, 1) (200, 2) (300, 3)
  BOOST_CHECK(vec.size() == 3); 
  BOOST_CHECK(vec[0][0] == 100);
  BOOST_CHECK(vec[0][1] == 1);
  BOOST_CHECK(vec[1][0] == 200);
  BOOST_CHECK(vec[1][1] == 2);
  BOOST_CHECK(vec[2][0] == 300);
  BOOST_CHECK(vec[2][1] == 3);
  
  std::vector<boost::aura::svec<std::size_t, 1> > vec2 = 
    generate_sequence<std::size_t, 1>("(100,200,300)");
  BOOST_CHECK(vec2[0][0] == 100);
  BOOST_CHECK(vec2[1][0] == 200);
  BOOST_CHECK(vec2[2][0] == 300);
}
