#define BOOST_TEST_MODULE device_allocator
#include <boost/test/unit_test.hpp>

#include <boost/aura/device_allocator.hpp>
#include <boost/aura/environment.hpp>

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic_allocator)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                d.allocation_tracker.activate();
                boost::aura::device_allocator<float> a(d);
                BOOST_CHECK(d.allocation_tracker.count_active() == 0);
                {
                        auto ptr = a.allocate(1024);
                        BOOST_CHECK(d.allocation_tracker.count_active() == 1);
                        a.deallocate(ptr, 1024);
                }
                BOOST_CHECK(d.allocation_tracker.count_active() == 0);
                BOOST_CHECK(d.allocation_tracker.count_old() == 1);

        }
        boost::aura::finalize();
}
