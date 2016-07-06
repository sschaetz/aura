// Work around a compiler bug on MacOS on Boost 1.60
#define BOOST_NO_CXX11_VARIADIC_TEMPLATES

#include <boost/aura/copy.hpp>
#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>
#include <boost/aura/environment.hpp>
#include <boost/aura/feed.hpp>
#include <boost/aura/invoke.hpp>
#include <boost/aura/kernel.hpp>
#include <boost/aura/library.hpp>

#include <test/test.hpp>

// This test is used to catch potential linker errors
// caused by duplicated symbols (forgot to inline).


void test_multi_comp_units2()
{
        boost::aura::initialize();
        {
                boost::aura::device d(AURA_UNIT_TEST_DEVICE);
                boost::aura::feed f(d);
                boost::aura::library l(
                        boost::aura::path(boost::aura::test::get_test_dir() +
                                "/kernels.al"),
                        d);
                boost::aura::kernel k("add", l);
                const std::size_t num_el = 128;

                std::vector<float> a(num_el, 2.0f);
                std::vector<float> b(num_el, 3.0f);
                std::vector<float> c(num_el, 0.0f);
                std::vector<float> expected(num_el, 2.0f + 3.0f);

                boost::aura::device_array<float> a_device(num_el, d);
                boost::aura::device_array<float> b_device(num_el, d);
                boost::aura::device_array<float> c_device(num_el, d);

                boost::aura::copy(a.begin(), a.end(), a_device.begin(), f);
                boost::aura::copy(b.begin(), b.end(), b_device.begin(), f);
                boost::aura::copy(c.begin(), c.end(), c_device.begin(), f);

                boost::aura::invoke(k, boost::aura::mesh({128, 1, 1}),
                        boost::aura::bundle({
                                1, 1, 1,
                        }),
                        boost::aura::args(a_device.get_base_ptr(),
                                            b_device.get_base_ptr(),
                                            c_device.get_base_ptr()),
                        f);

                boost::aura::copy(
                        c_device.begin(), c_device.end(), c.begin(), f);
                boost::aura::wait_for(f);
        }
        boost::aura::finalize();
        return;
}
