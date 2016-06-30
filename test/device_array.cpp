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
                BOOST_CHECK(ar1.size() == 1024);
                device_array<float> ar2(bounds({2, 2, 2, 2}), d);
                BOOST_CHECK(ar2.size() == 2 * 2 * 2 * 2);
                BOOST_CHECK(ar2.bounds() == bounds({2, 2, 2, 2}));
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

BOOST_AUTO_TEST_CASE(iterators)
{
        initialize();
        {
                device d(AURA_UNIT_TEST_DEVICE);
                device_array<float> ar0(bounds({2, 2, 2, 2}), d);
                auto begin = ar0.begin();
                auto end = ar0.end();
                BOOST_CHECK(begin != end);
                BOOST_CHECK(begin + product(bounds({2, 2, 2, 2})) == end);
                BOOST_CHECK(ar0.begin() != ar0.end());
                auto end2 = ar0.begin() + product(bounds({2, 2, 2, 2}));
                BOOST_CHECK(ar0.begin() + product(bounds({2, 2, 2, 2})) ==
                        ar0.end());
        }
        finalize();
}
