#define BOOST_TEST_MODULE device_array

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
                boost::ignore_unused(end2);
                BOOST_CHECK(ar0.begin() + product(bounds({2, 2, 2, 2})) ==
                        ar0.end());
        }
        finalize();
}

BOOST_AUTO_TEST_CASE(basic_copy)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);

                std::vector<float> host_src(1024, 21.0f);
                std::vector<float> host_dst(1024, 0.0f);

                boost::aura::device_array<float> src(1024, d);
                boost::aura::device_array<float> dst(1024, d);

                // Host to device
                boost::aura::copy(
                        host_src.begin(), host_src.end(), src.begin(), f);
                // Device to device.
                boost::aura::copy(src.begin(), src.end(), dst.begin(), f);
                // Device to host
                boost::aura::copy(dst.begin(), dst.end(), host_dst.begin(), f);

                boost::aura::wait_for(f);
                BOOST_CHECK(std::equal(
                        host_src.begin(), host_src.end(), host_dst.begin()));
        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(shared)
{
        initialize();
        {
                device d(AURA_UNIT_TEST_DEVICE);
                device_array<float> ar0(bounds({2, 2, 2, 2}), d);
#ifdef AURA_BASE_METAL
                BOOST_CHECK(ar0.is_shared_memory() == true);
#else
                BOOST_CHECK(ar0.is_shared_memory() == false);
#endif
        }
        finalize();
}

BOOST_AUTO_TEST_CASE(basic_zero)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);

                std::vector<float> host_src(1024, 21.0f);
                std::vector<float> host_dst(1024, 42.0f);
                std::vector<float> expected(1024, 0.0f);

                boost::aura::device_array<float> dev(1024, d);

                // Host to device
                boost::aura::copy(
                        host_src.begin(), host_src.end(), dev.begin(), f);

                // Zero device.
                dev.zero(f);

                // Device to host
                boost::aura::copy(dev.begin(), dev.end(), host_dst.begin(), f);

                boost::aura::wait_for(f);

                BOOST_CHECK(std::equal(
                        expected.begin(), expected.end(), host_dst.begin()));
        }
        boost::aura::finalize();
}
