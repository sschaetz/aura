#define BOOST_TEST_MODULE device_array

#include <algorithm>

#include <boost/test/unit_test.hpp>

#include <boost/aura/copy.hpp>
#include <boost/aura/device.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/environment.hpp>

#include <test/test.hpp>

#include <boost/core/ignore_unused.hpp>

using namespace boost::aura;

// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(basic)
{
        initialize();
        {
                device d(AURA_UNIT_TEST_DEVICE);
                feed f(d);
                device_array<float> ar0(1024, d);
                ar0.zero(f);
                {
                        auto m0 = ar0.map(f);
                        f.synchronize();
                        // Fill map with values.
                        std::fill(m0.begin(), m0.end(), 42.0f);
                }
                std::vector<float> vec(1024, 0.0f);
                boost::aura::copy(ar0, vec, f);
                f.synchronize();
                BOOST_CHECK(std::all_of(vec.begin(), vec.end(), [](float v) { return v == 42.0f; } ));
        }
        finalize();
}
