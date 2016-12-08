#define BOOST_TEST_MODULE device
#include <boost/test/unit_test.hpp>

#include <boost/aura/device.hpp>
#include <boost/aura/environment.hpp>
#include <boost/aura/feed.hpp>

#include <boost/core/ignore_unused.hpp>

#include <iostream>

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(empty_device)
{
        boost::aura::initialize();
        {
                boost::aura::device d;
        }
        boost::aura::finalize();
}

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

BOOST_AUTO_TEST_CASE(basic_num_devices)
{
        boost::aura::initialize();
        {
                std::cout << "Num devices in system: " <<
                        boost::aura::device::num() <<
                        std::endl;
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
                boost::ignore_unused(base_device_handle);
#ifndef AURA_BASE_METAL
                auto base_context_handle = d.get_base_context();
                boost::ignore_unused(base_context_handle);
#endif
                BOOST_CHECK(d.get_ordinal() == AURA_UNIT_TEST_DEVICE);
        }
        boost::aura::finalize();
}


// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(device_ctors)
{
        boost::aura::initialize();
        {
                boost::aura::device d0;
                d0 = boost::aura::device(AURA_UNIT_TEST_DEVICE);
                boost::aura::device d1(std::move(d0));
                boost::aura::device d2(AURA_UNIT_TEST_DEVICE);
                d2 = std::move(d1);
        }
        boost::aura::finalize();
}
