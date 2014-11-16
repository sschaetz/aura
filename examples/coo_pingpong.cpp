#include <algorithm>
#include <vector>
#include <complex>
#include <boost/aura/backend.hpp>
#include <boost/aura/bounds.hpp>
#include <boost/aura/misc/coo.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/copy.hpp>

typedef std::complex<float> cfloat;
using namespace boost::aura;

void generate_data()
{
	boost::aura::bounds b(16, 16, 1, 1, 10);
	std::vector<cfloat> hv(boost::aura::product(b), cfloat(42.0, 21.0));
	boost::aura::coo_write(hv.begin(), b, "/tmp/mytest.coo");
}

int main(void)
{
	initialize();
	generate_data();

	// -----

	device d(0);
	feed f(d);

	// load coo
	boost::aura::bounds b;
	std::vector<std::complex<float>> hv0;
	std::tie(hv0, b) = boost::aura::coo_read<cfloat>("/tmp/mytest.coo");
	std::cout << "coo loaded, dimensions: " << b << std::endl;

	// upload to device, copy, download from device
	boost::aura::device_array<cfloat> dv0(b, d);
	boost::aura::device_array<cfloat> dv1(b, d);

	boost::aura::copy(hv0, dv0, f);
	boost::aura::copy(dv0, dv1, f);

	std::vector<cfloat> hv1(boost::aura::product(b), cfloat(0.0, 0.0));
	boost::aura::copy(dv1, hv1, f);
	boost::aura::wait_for(f);

	// write result
	boost::aura::coo_write(hv1.begin(), b, "/tmp/mytest2.coo");
}

