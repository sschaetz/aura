
#include <complex>
#include <vector>
#include <stdio.h>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;

const char* atomic_cfloat_kernel_str = R"kernel_str(
	
	#include <boost/aura/backend.hpp>

	AURA_KERNEL void atomic_cfloat(AURA_GLOBAL cfloat* src)
	{
		// real
		AURA_GLOBAL float* re = src;
		// imaginary
		AURA_GLOBAL float* im = src;
		im++;
		
		// atomic add
		atomic_addf(re, 1.0);
		atomic_addf(im, 1.0);

	}
		
		)kernel_str";

int main(void) 
{
	initialize();
	device d(0);  
	feed f(d);
	auto atomic_cfloat = d.load_from_string(
			"atomic_cfloat",
			atomic_cfloat_kernel_str,
			AURA_BACKEND_COMPILE_FLAGS, true);
	std::vector<std::complex<float>> v(1);
	device_array<std::complex<float>> dv(1, d);
	copy(v, dv, f);
	invoke(atomic_cfloat, bounds(100000), args(dv.begin_ptr()), f);
	copy(dv, v, f);
	wait_for(f);
	std::cout << v[0] << std::endl;
}


