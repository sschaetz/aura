#define BOOST_TEST_MODULE preprocessor

#include <algorithm>
#include <vector>
#include <complex>
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/aura/preprocessor.hpp>

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic)
{
	{
		std::string input("foo <<<foo>>> bar");
		boost::aura::preprocessor p;

		float f = 5.0;
		p.add_define("foo", f);
		auto result = p(input);
		BOOST_CHECK(result == "foo 5.0000000f bar");
	}
	{
		std::string input("foo <<<foo>>> bar");
		boost::aura::preprocessor p;

		std::size_t f = 5;
		p.add_define("foo", f);
		auto result = p(input);
		BOOST_CHECK(result == "foo 5 bar");
	}
	{
		std::string input("foo <<<foo>>> bar");
		boost::aura::preprocessor p;

		std::vector<double> v({1.0000001, 2.0});
		p.add_define("foo", v);
		auto result = p(input);
		BOOST_CHECK(result == "foo {1.0000001,2} bar");
	}
}
