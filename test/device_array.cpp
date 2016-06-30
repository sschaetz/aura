#define BOOST_TEST_MODULE device_array

#include <boost/test/unit_test.hpp>

#include <boost/aura/device.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/environment.hpp>

#include <test/test.hpp>

using namespace boost::aura;

// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(basic)
{
        initialize();
        {
                device d(AURA_UNIT_TEST_DEVICE);
                device_array<float> ar0;
                device_array<float> ar1(1024, d);
                device_array<float> ar2(bounds({2, 2, 2, 2}), d);
        }
        finalize();
}

BOOST_AUTO_TEST_CASE(move)
{
        initialize();
        {
                device d(AURA_UNIT_TEST_DEVICE);
                device_array<float> ar0(bounds({2, 2, 2, 2}), d);
                auto ar1 = std::move(ar0);
                auto ar2(std::move(ar1));
        }
        finalize();
}
