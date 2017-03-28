#define BOOST_TEST_MODULE device_allocator
#include <boost/test/unit_test.hpp>

#include <boost/aura/device_allocator.hpp>
#include <boost/aura/device_pool_allocator.hpp>
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

BOOST_AUTO_TEST_CASE(basic_pool_allocator)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                d.allocation_tracker.activate();
                {
                        boost::aura::device_pool_allocator<float> a(d);
                        BOOST_CHECK(d.allocation_tracker.count_active() == 0);
                        {
                                auto ptr = a.allocate(1024);
                                BOOST_CHECK(
                                        d.allocation_tracker.count_active() ==
                                        1
                                );

                                a.deallocate(ptr, 1024);
                        }
                }
                BOOST_CHECK(d.allocation_tracker.count_active() == 0);
                BOOST_CHECK(d.allocation_tracker.count_old() == 1);

        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(pool_allocator_funcionality)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                d.allocation_tracker.activate();
                {
                        boost::aura::device_pool_allocator<float> a(d);
                        BOOST_CHECK(d.allocation_tracker.count_active() == 0);
                        {
                                for (int i = 0; i<100; i++)
                                {
                                        auto ptr = a.allocate(1024);
                                        a.deallocate(ptr, 1024);
                                }
                                // Make sure we only allocated once.
                                BOOST_CHECK(
                                        d.allocation_tracker.count_active() ==
                                        1
                                );
                                BOOST_CHECK(
                                        d.allocation_tracker.count_old() ==
                                        0
                                );
                        }
                }
                BOOST_CHECK(d.allocation_tracker.count_active() == 0);
                BOOST_CHECK(d.allocation_tracker.count_old() == 1);

        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(pool_allocator_multiple_sizes)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                d.allocation_tracker.activate();
                {
                        boost::aura::device_pool_allocator<float> a(d);
                        BOOST_CHECK(d.allocation_tracker.count_active() == 0);
                        {
                                for (int i = 0; i<100; i++)
                                {
                                        auto ptr0 = a.allocate(1024);
                                        auto ptr1 = a.allocate(2*1024);
                                        a.deallocate(ptr0, 1024);
                                        a.deallocate(ptr1, 2*1024);
                                }
                                // Make sure we only allocated once.
                                BOOST_CHECK(
                                        d.allocation_tracker.count_active() ==
                                        2
                                );
                                BOOST_CHECK(
                                        d.allocation_tracker.count_old() ==
                                        0
                                );
                        }
                }
                BOOST_CHECK(d.allocation_tracker.count_active() == 0);
                BOOST_CHECK(d.allocation_tracker.count_old() == 2);

        }
        boost::aura::finalize();
}
