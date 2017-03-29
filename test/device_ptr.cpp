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
                boost::aura::device_ptr<float> ptr0;
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
                BOOST_CHECK(ptr - ptr == 0);
                BOOST_CHECK(ptr + ptr == 0);
                auto ptr2 = ptr + 100;
                BOOST_CHECK(ptr2 - ptr == 100);
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

BOOST_AUTO_TEST_CASE(shared)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                auto ptr =
                        boost::aura::device_malloc<float>(1024 * 1024 * 20, d);
#ifdef AURA_BASE_METAL
                BOOST_CHECK(ptr.is_shared_memory() == true);
#else
                BOOST_CHECK(ptr.is_shared_memory() == false);
#endif
                boost::aura::device_free(ptr);
        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(less_comparison)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::device_ptr<float> ptr_null0;
                boost::aura::device_ptr<float> ptr_null1;

                auto ptr0 = boost::aura::device_malloc<float>(4, d);

                BOOST_CHECK(!(ptr_null0 < ptr_null1));
                BOOST_CHECK(!(ptr_null1 < ptr_null0));

                BOOST_CHECK(ptr_null0 < ptr0);
                BOOST_CHECK(!(ptr0 < ptr_null0));

                auto ptr1 = ptr0;
                BOOST_CHECK(!(ptr0 < ptr1));
                BOOST_CHECK(!(ptr1 < ptr0));

                ptr1 += 5;
                BOOST_CHECK(ptr0 < ptr1);
                BOOST_CHECK(!(ptr1 < ptr0));

                boost::aura::device_free(ptr0);
        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(hash)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::device_ptr<float> ptr_null0;
                boost::aura::device_ptr<float> ptr_null1;

                BOOST_CHECK(
                        std::hash<boost::aura::device_ptr<float>>()(ptr_null0)
                        ==
                        std::hash<boost::aura::device_ptr<float>>()(ptr_null0)
                );

                BOOST_CHECK(
                        std::hash<boost::aura::device_ptr<float>>()(ptr_null0)
                        ==
                        std::hash<boost::aura::device_ptr<float>>()(ptr_null1)
                );

                auto ptr0 = boost::aura::device_malloc<float>(4, d);

                BOOST_CHECK(
                        std::hash<boost::aura::device_ptr<float>>()(ptr_null0)
                        !=
                        std::hash<boost::aura::device_ptr<float>>()(ptr0)
                );

                auto ptr1 = ptr0;

                BOOST_CHECK(
                        std::hash<boost::aura::device_ptr<float>>()(ptr1)
                        ==
                        std::hash<boost::aura::device_ptr<float>>()(ptr0)
                );

                ptr1 += 5;

                BOOST_CHECK(
                        std::hash<boost::aura::device_ptr<float>>()(ptr1)
                        !=
                        std::hash<boost::aura::device_ptr<float>>()(ptr0)
                );

                boost::aura::device_free(ptr0);
        }
        boost::aura::finalize();
}
