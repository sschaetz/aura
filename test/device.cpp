#define BOOST_TEST_MODULE device
#include <boost/test/unit_test.hpp>

#include <boost/aura/environment.hpp>
#include <boost/aura/device.hpp>

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic)
{
        boost::aura::initialize();
        boost::aura::device d(AURA_UNIT_TEST_DEVICE);
}

