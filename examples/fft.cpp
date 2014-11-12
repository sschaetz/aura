#include <algorithm>
#include <vector>
#include <complex>
#include <aura/backend.hpp>
#include <aura/bounds.hpp>
#include <aura/misc/coo.hpp>
#include <aura/device_array.hpp>
#include <aura/copy.hpp>
#include <aura/fft.hpp>

typedef std::complex<float> cfloat;
using namespace aura;

int main(void)
{
	initialize();
	// -----

	device d(0);
	feed f(d);

	// load coo
	bounds b;
	std::vector<std::complex<float>> hv0;
	std::tie(hv0, b) = coo_read<cfloat>("/tmp/T3519.med");
	std::cout << "coo loaded, dimensions: " << b << std::endl;

	// upload to device
	device_array<cfloat> dv0(b, d);
	device_array<cfloat> dv1(b, d);

	copy(hv0, dv0, f);

	// calculate inverse transform
	fft_initialize();
	fft fh(d, f, b, fft::type::c2c);
	fft_inverse(dv0, dv1, fh, f);
	fft_terminate();
		
	std::vector<cfloat> hv1(product(b), cfloat(0.0, 0.0));
	copy(dv1, hv1, f);
	wait_for(f);

	// write result
	coo_write(hv1.begin(), b, "/tmp/T3519.coo");
}

