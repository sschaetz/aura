#define BOOST_TEST_MODULE io

#include <boost/aura/io.hpp>
#include <test/test.hpp>

#include <iostream>

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(test_read_all)
{
        std::string contents = boost::aura::read_all(boost::aura::path(
                boost::aura::test::get_test_dir() + "/io.txt"));

        std::string expected =
                R"(Though this be madness, yet there is method in 't.)"
                "\n"
                R"(Brevity is the soul of wit.)"
                "\n";

        BOOST_CHECK(expected == contents);
}
