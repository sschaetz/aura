#define BOOST_TEST_MODULE basic
#include <boost/test/unit_test.hpp>

#include <boost/aura/device.hpp>
#include <boost/aura/environment.hpp>
//#include <boost/aura/feed.hpp>


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
                //auto base_device_handle = d.get_base_device();
                //auto base_context_handle = d.get_base_context();
		//BOOST_CHECK(d.get_ordinal() == AURA_UNIT_TEST_DEVICE);
        }
        boost::aura::finalize();
}

/*
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(basic_feed)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);
        }
        boost::aura::finalize();
}
*/
