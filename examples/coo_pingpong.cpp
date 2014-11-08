#include <algorithm>
#include <vector>
#include <complex>
#include <aura/backend.hpp>
#include <aura/bounds.hpp>
#include <aura/misc/coo.hpp>
#include <aura/device_array.hpp>
#include <aura/copy.hpp>

typedef std::complex<float> cfloat;
using namespace aura;

void generate_data()
{
	aura::bounds b(16, 16, 1, 1, 10);
	std::vector<cfloat> hv(aura::product(b), cfloat(42.0, 21.0));
	aura::coo_write(hv.begin(), b, "/tmp/mytest.coo");
}

int main(void)
{
	generate_data();

	// -----

	device d(0);
	feed f(d);

	// load coo
	aura::bounds b;
	std::vector<std::complex<float>> hv0;
	std::tie(hv0, b) = aura::coo_read<cfloat>("/tmp/mytest.coo");
	std::cout << "coo loaded, dimensions: " << b << std::endl;

	// upload to device, copy, download from device
	aura::device_array<cfloat> dv0(b, d);
	aura::device_array<cfloat> dv1(b, d);

	aura::copy(hv0, dv0, f);
	aura::copy(dv0, dv1, f);

	std::vector<cfloat> hv1(aura::product(b), cfloat(0.0, 0.0));
	aura::copy(dv1, hv1, f);
	aura::wait_for(f);

	// write result
	aura::coo_write(hv1.begin(), b, "/tmp/mytest2.coo");
}

