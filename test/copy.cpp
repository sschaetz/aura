#define BOOST_TEST_MODULE copy_test
#include <boost/test/unit_test.hpp>

#include <boost/aura/copy.hpp>
#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>
#include <boost/aura/environment.hpp>
#include <boost/aura/feed.hpp>

// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(excessive_ptr)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);

                std::vector<float> host_src(1024, 21.0f);
                std::vector<float> host_dst(1024, 0.0f);

                auto ptr = boost::aura::device_malloc<float>(1024, d);
                boost::aura::copy(host_src.begin(), host_src.end(), ptr, f);
                boost::aura::copy(ptr, ptr+1024, host_dst.begin(), f);
                boost::aura::wait_for(f);
                boost::aura::device_free(ptr);
                BOOST_CHECK(std::equal(host_src.begin(), host_src.end(), host_dst.begin()));
        }
        boost::aura::finalize();
}
