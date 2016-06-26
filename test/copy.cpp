#define BOOST_TEST_MODULE copy_test
#include <boost/test/unit_test.hpp>

#include <boost/aura/copy.hpp>
#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>
#include <boost/aura/environment.hpp>
#include <boost/aura/feed.hpp>

// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(basic_copy)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);

                std::vector<float> host_src(1024, 21.0f);
                std::vector<float> host_dst(1024, 0.0f);

                auto ptr0 = boost::aura::device_malloc<float>(1024, d);
                auto ptr1 = boost::aura::device_malloc<float>(1024, d);

                // Host to device
                boost::aura::copy(host_src.begin(), host_src.end(), ptr0, f);
                // Device to device.
                boost::aura::copy(ptr0, ptr0 + 1024, ptr1, f);
                // Device to host
                boost::aura::copy(ptr1, ptr1 + 1024, host_dst.begin(), f);

                boost::aura::wait_for(f);
                boost::aura::device_free(ptr0);
                boost::aura::device_free(ptr1);
                BOOST_CHECK(std::equal(
                        host_src.begin(), host_src.end(), host_dst.begin()));
        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(out_of_bounds_copy)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);

                std::vector<float> host_src(1024, 21.0f);
                std::vector<float> host_dst(1024 + 16, 0.0f);

                auto ptr0 = boost::aura::device_malloc<float>(1024, d);

                // Host to device
                boost::aura::copy(host_src.begin(), host_src.end(), ptr0, f);
                // Device to host
                boost::aura::copy(ptr0, ptr0 + 1024, host_dst.begin(), f);

                boost::aura::wait_for(f);
                boost::aura::device_free(ptr0);
                BOOST_CHECK(std::equal(
                        host_src.begin(), host_src.end(), host_dst.begin()));
                for (auto it = host_dst.begin() + 1024; it != host_dst.end();
                        it++)
                {
                        if (*it != 0.0f)
                        {
                                BOOST_ERROR("Out of bounds copy detected.");
                        }
                }
        }
        boost::aura::finalize();
}
