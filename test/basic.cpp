#define BOOST_TEST_MODULE basic
#include <boost/test/unit_test.hpp>

#include <boost/aura/device.hpp>
#include <boost/aura/environment.hpp>
#include <boost/aura/feed.hpp>


// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic_device)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
        }
        boost::aura::finalize();
}

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic_device_getters)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                auto base_device_handle = d.get_base_device();
                auto base_context_handle = d.get_base_context();
		BOOST_CHECK(d.get_ordinal() == AURA_UNIT_TEST_DEVICE);
        }
        boost::aura::finalize();
}


// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(basic_feed)
{
        boost::aura::initialize();
        {
                boost::aura::feed f0();
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);
                f.synchronize();
                wait_for(f);
                auto base_device_handle = f.get_base_device();
                auto base_context_handle = f.get_base_context();
                auto base_feed_handle = f.get_base_feed();
		BOOST_CHECK(f.get_device().get_ordinal() == AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f2(std::move(f));
                f = std::move(f2);

        }
        boost::aura::finalize();
}

