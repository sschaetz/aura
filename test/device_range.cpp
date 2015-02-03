#define BOOST_TEST_MODULE device_range

#include <boost/test/unit_test.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/device_range.hpp>

using namespace boost::aura;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) 
{
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);
		device_array<int> array1(40, d);
		device_range<int> range1(array1, slice(5));
		auto range2 = array1[slice(5)];

		BOOST_CHECK(range1.begin() == range2.begin());
		BOOST_CHECK(range1.size() == range2.size());
		BOOST_CHECK(range1.size() == 1);
		BOOST_CHECK(range1.get_bounds() == range2.get_bounds());
	}
}

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(copy_test) 
{
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);
		feed f(d);
		device_array<int> array1(bounds(40, 35, 10), d);
		std::vector<int> vec1(array1.size(), 0);
		copy(vec1, array1, f);
		std::vector<int> vec2(40, 1);
		auto range1 = array1[slice(_, 1, 1)];
		BOOST_CHECK(range1.size() == vec2.size());
		copy(vec2, range1, f);
		copy(array1, vec1, f);
		wait_for(f);
		for (std::size_t i=0; i<array1.size(); i++) {
			if (i <40*35+40) {
				BOOST_CHECK(vec1[i] == 0);
			} else if (i < 40*35+40+40) {
				BOOST_CHECK(vec1[i] == 1);
			} else {
				BOOST_CHECK(vec1[i] == 0);
			}
		}
	}
}

