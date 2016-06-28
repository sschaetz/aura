#define BOOST_TEST_MODULE tiny_vector

#include <boost/aura/bounds/tiny_vector.hpp>
#include <boost/aura/bounds/product.hpp>

#include <boost/test/unit_test.hpp>

using namespace boost::aura;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(test_basic)
{
        using bounds = tiny_vector<std::size_t, 10>;
        bounds b0;
        BOOST_CHECK(b0.size() == 0);
        bounds b1({0, 1, 2, 3, 4, 5, 6, 7});
        BOOST_CHECK(b1.size() == 8);
}

BOOST_AUTO_TEST_CASE(test_product)
{
        using bounds = tiny_vector<std::size_t, 10>;
        bounds b0;
        BOOST_CHECK(product(b0) == 0);
        bounds b1({1, 2, 3, 4, 5, 6, 7, 8, 9});
        BOOST_CHECK(product(b1) == 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9);
        BOOST_CHECK((std::size_t)(b1) == 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9);
}

BOOST_AUTO_TEST_CASE(test_other)
{
        using bounds = tiny_vector<std::size_t, 10>;
        bounds b0({1, 2, 3, 4, 5, 6, 7, 8, 9});
        b0.debug__();
        bounds b1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        BOOST_CHECK(b0 != b1);
        BOOST_CHECK(b0 == b0);
        b1.pop_back();
        BOOST_CHECK(b0 == b1);
}
