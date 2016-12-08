#define BOOST_TEST_MODULE feed
#include <boost/test/unit_test.hpp>

#include <boost/aura/device.hpp>
#include <boost/aura/environment.hpp>
#include <boost/aura/feed.hpp>

#include <boost/core/ignore_unused.hpp>

#include <iostream>

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic_feed)
{
        boost::aura::initialize();
        {
                boost::aura::feed f0;
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);
                f.synchronize();
                boost::aura::wait_for(f);
                auto base_device_handle = f.get_base_device();
                boost::ignore_unused(base_device_handle);
#ifndef AURA_BASE_METAL
                auto base_context_handle = f.get_base_context();
                boost::ignore_unused(base_context_handle);
#endif
                auto base_feed_handle = f.get_base_feed();
                boost::ignore_unused(base_feed_handle);
                BOOST_CHECK(
                        f.get_device().get_ordinal() == AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f2(std::move(f));
                f = std::move(f2);

#ifdef AURA_BASE_METAL
                auto command_buffer = f.get_command_buffer();
                boost::ignore_unused(command_buffer);
#endif
        }
        boost::aura::finalize();
}
