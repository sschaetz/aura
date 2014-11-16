#define BOOST_TEST_MODULE backend.p2p

#include <vector>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <boost/aura/backend.hpp>

using namespace boost::aura::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
	initialize();
	std::vector<int> mtrx = get_peer_access_matrix();
}


