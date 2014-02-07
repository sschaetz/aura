
#include <complex>
#include <cufft.h>
#include <aura/backend.hpp>
#include <aura/fft.hpp>
#include <aura/device_array.hpp>
#include <aura/misc/benchmark.hpp>

using namespace aura;

typedef std::complex<float> cfloat;

/// benchmark 2d fft vs 1d fft performance

// run each test for a specific number of seconds
const int duration_per_test = 2*1e6;

int main(void) 
{
	initialize();
	int num = device_get_count();
	if(1 >= num) {
		printf("no devices found\n"); exit(0);
	}
	device d(0);
	feed f(d);

	int dim1 = 256;
	int dim2 = 256;

	device_array<std::complex<float> > src(bounds(dim1, dim2), d);
	device_array<std::complex<float> > dst(bounds(dim1, dim2), d);

	fft plan2d(d, f, bounds(dim1, dim2), fft::c2c);
	fft plan1d(d, f, bounds(dim1), fft::c2c, dim2);

	double min, max, mean, stdev;
	std::size_t runs;
	
	AURA_BENCHMARK_ASYNC(fft_forward(src, dst, plan2d, f), 
			wait_for(f), 
			duration_per_test, min, max, mean, stdev, runs);
	printf("%s: runs %lu min %f max %f mean %f stdev %f\n", 
		"plan2d", runs, min, max, mean, stdev);
	
	AURA_BENCHMARK_ASYNC(fft_forward(src, dst, plan1d, f), 
			wait_for(f),
			duration_per_test, min, max, mean, stdev, runs);
	printf("%s: runs %lu min %f max %f mean %f stdev %f\n", 
		"plan1d", runs, min, max, mean, stdev);
	


}

