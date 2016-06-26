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
                boost::aura::device_free(ptr);
        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(pointer_arithmetic)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                auto ptr = boost::aura::device_malloc<float>(1024, d);
                BOOST_CHECK(ptr-ptr == 0);
                BOOST_CHECK(ptr+ptr == 0);
                auto ptr2 = ptr + 100;
                BOOST_CHECK(ptr2-ptr == 100);
                boost::aura::device_free(ptr);
        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(excessive_ptr)
{
        boost::aura::initialize();
        {
                for (unsigned int i = 0; i < 5; i++)
                {
                        boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                        auto ptr = boost::aura::device_malloc<float>(
                                1024 * 1024 * 20, d);
                        boost::aura::device_free(ptr);
                }
        }
        boost::aura::finalize();
}
