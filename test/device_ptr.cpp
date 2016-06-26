#define BOOST_TEST_MODULE basic
#include <boost/test/unit_test.hpp>

#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>
#include <boost/aura/environment.hpp>

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic_ptr)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                auto ptr = boost::aura::device_malloc<float>(1024, d);
                device_free(ptr);
        }
        boost::aura::finalize();
}
