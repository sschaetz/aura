#define BOOST_TEST_MODULE device_array

#include <boost/test/unit_test.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);
		device_array<int> array1(40, d);
		device_array<int> array2(bounds(40, 20, 10), d);
		BOOST_CHECK(40 == array1.size());
		BOOST_CHECK(40*20*10 == array2.size());
	}
}

// resize
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(resize) {
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);
		device_array<int> array1(40, d);
		device_array<int> array2(bounds(40, 20, 10), d);
		BOOST_CHECK(40 == array1.size());
		BOOST_CHECK(40*20*10 == array2.size());

		array1.resize(bounds(20, 20, 100));
		BOOST_CHECK(20*20*100 == array1.size());

		array1.resize(bounds(20, 20, 200), d);
		BOOST_CHECK(20*20*200 == array1.size());

		array1.resize(5);
		BOOST_CHECK(5 == array1.size());

		array1.resize(6, d);
		BOOST_CHECK(6 == array1.size());

	}
}

// iterator interface
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(iteratorinterface) {
	initialize();
	int num = device_get_count();
	if(0 < num) {
		device d(0);
		feed f(d);
		std::vector<std::size_t> b = {23, 4, 2, 7};
		std::size_t s = std::accumulate(b.begin(), b.end(), 1,
				std::multiplies<std::size_t>());

		std::vector<float> data(s, 15.0);

		device_array<float> device_data(data.begin(), b, d, f);

		std::vector<float> result(s, 0.0);
		copy(device_data, result, f);
		wait_for(f);
		BOOST_CHECK(std::equal(result.begin(), result.end(),
					data.begin()));
	}
}

