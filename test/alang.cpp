#define BOOST_TEST_MODULE library
#include <boost/test/unit_test.hpp>

#include <boost/aura/copy.hpp>
#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>
#include <boost/aura/environment.hpp>
#include <boost/aura/feed.hpp>
#include <boost/aura/invoke.hpp>
#include <boost/aura/kernel.hpp>
#include <boost/aura/library.hpp>

#include <test/test.hpp>

#include <iostream>


// _____________________________________________________________________________


BOOST_AUTO_TEST_CASE(basic_alang)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);
                boost::aura::library l(
                        boost::aura::path(boost::aura::test::get_test_dir() +
                                "/kernels.al"),
                        d);
                boost::aura::kernel k("add_alang", l);
                const std::size_t num_el = 128;

                std::vector<float> a(num_el, 2.0f);
                std::vector<float> b(num_el, 3.0f);
                std::vector<float> c(num_el, 0.0f);
                std::vector<float> expected(num_el, 2.0f + 3.0f);

                auto a_ptr = boost::aura::device_malloc<float>(num_el, d);
                auto b_ptr = boost::aura::device_malloc<float>(num_el, d);
                auto c_ptr = boost::aura::device_malloc<float>(num_el, d);

                boost::aura::copy(a.begin(), a.end(), a_ptr, f);
                boost::aura::copy(b.begin(), b.end(), b_ptr, f);
                boost::aura::copy(c.begin(), c.end(), c_ptr, f);

                boost::aura::invoke(k, boost::aura::mesh({{128, 1, 1}}),
                        boost::aura::bundle({{1, 1, 1}}),
                        boost::aura::args(a_ptr.get_base_ptr(),
                                b_ptr.get_base_ptr(), c_ptr.get_base_ptr()),
                        f);

                boost::aura::copy(c_ptr, c_ptr + num_el, c.begin(), f);
                boost::aura::wait_for(f);

                BOOST_CHECK(std::equal(
                        expected.begin(), expected.end(), c.begin()));
        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(alang_all_mesh)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);
                boost::aura::library l(
                        boost::aura::path(boost::aura::test::get_test_dir() +
                                "/kernels.al"),
                        d);
                boost::aura::kernel k("all_alang_mesh", l);
                const std::size_t num_el = 8 * 9 * 15;

                std::vector<float> a(num_el, 2.0f);
                std::vector<float> b(num_el, 3.0f);
                std::vector<float> c(num_el, 0.0f);
                std::vector<float> expected(num_el, 2.0f + 3.0f);

                auto a_ptr = boost::aura::device_malloc<float>(num_el, d);
                auto b_ptr = boost::aura::device_malloc<float>(num_el, d);
                auto c_ptr = boost::aura::device_malloc<float>(num_el, d);

                boost::aura::copy(a.begin(), a.end(), a_ptr, f);
                boost::aura::copy(b.begin(), b.end(), b_ptr, f);
                boost::aura::copy(c.begin(), c.end(), c_ptr, f);

                boost::aura::invoke(k, boost::aura::mesh({{8, 9, 15}}),
                        boost::aura::bundle({{2, 3, 5}}),
                        boost::aura::args(a_ptr.get_base_ptr(),
                                b_ptr.get_base_ptr(), c_ptr.get_base_ptr()),
                        f);

                boost::aura::copy(c_ptr, c_ptr + num_el, c.begin(), f);
                boost::aura::wait_for(f);

                BOOST_CHECK(std::equal(
                        expected.begin(), expected.end(), c.begin()));
        }
        boost::aura::finalize();
}

BOOST_AUTO_TEST_CASE(alang_all_bundle)
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);
                boost::aura::library l(
                        boost::aura::path(boost::aura::test::get_test_dir() +
                                "/kernels.al"),
                        d);
                boost::aura::kernel k("all_alang_bundle", l);
                const std::size_t num_el = 4 * 6 * 10;

                std::vector<float> a(num_el, 2.0f);
                std::vector<float> b(num_el, 3.0f);
                std::vector<float> c(num_el, 0.0f);
                std::vector<float> expected(num_el, 2.0f + 3.0f);

                auto a_ptr = boost::aura::device_malloc<float>(num_el, d);
                auto b_ptr = boost::aura::device_malloc<float>(num_el, d);
                auto c_ptr = boost::aura::device_malloc<float>(num_el, d);

                boost::aura::copy(a.begin(), a.end(), a_ptr, f);
                boost::aura::copy(b.begin(), b.end(), b_ptr, f);
                boost::aura::copy(c.begin(), c.end(), c_ptr, f);

                boost::aura::invoke(k, boost::aura::mesh({{4, 6, 10}}),
                        boost::aura::bundle({{4, 6, 10}}),
                        boost::aura::args(a_ptr.get_base_ptr(),
                                b_ptr.get_base_ptr(), c_ptr.get_base_ptr()),
                        f);

                boost::aura::copy(c_ptr, c_ptr + num_el, c.begin(), f);
                boost::aura::wait_for(f);

                BOOST_CHECK(std::equal(
                        expected.begin(), expected.end(), c.begin()));
        }
        boost::aura::finalize();
}
