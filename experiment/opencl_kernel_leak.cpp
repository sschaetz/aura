
#include <complex>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <boost/aura/backend.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/device_array.hpp>

using namespace boost::aura;

const char* empty_kernel_str = R"kernel_str(

	#include <boost/aura/backend.hpp>

	AURA_KERNEL void empty_kernel(AURA_GLOBAL cfloat* src)
	{
	}

		)kernel_str";

void test(void)
{
	initialize();
	device d(2);
	feed f(d);
	auto empty_kernel = d.load_from_string(
			"empty_kernel",
			empty_kernel_str,
			AURA_BACKEND_COMPILE_FLAGS, true);
	std::vector<std::complex<float>> v(1);
	device_array<std::complex<float>> dv(1, d);
	copy(v, dv, f);
	invoke(empty_kernel, bounds(100000), args(dv.data()), f);
	wait_for(f);
}


int main(void)
{
    test();
    // at this point, everything should have been freed correctly.
    // Checking the memory usage, though, reveals a leak
    std::cin.ignore();
}



