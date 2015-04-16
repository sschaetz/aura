// run copy and zero copy benchmarks

/**
 * results: CUDA normal Nvidia GPU
 *
 * compute only is really fast
 * compute copy the slowest
 * compute map is significantly (>2x) faster than compute copy
 * map only takes not time at all
 * compute only on a map is not as fast as compute only on an array
 *
 * results: OpenCL normal Nvidia GPU
 * compute only is as fast as CUDA
 * compute copy is the slowest
 * compute map is as slow as compute copy (-> map is emulated with copy)
 * compute only on a map is as fast as compute only on an array
 *
 * results: OpenCL on Odroid
 * compute only is fastest
 * compute copy is slowest
 * compute map is not as fast as it should be (far slower than compute only!)
 * map only is really fast
 * compute only + map only for this plattform is a lot faster
 * than compute and map combined, so either I'm doing something wrong or the
 * OpenCL implementation has a bug
 */

#include <vector>
#include <tuple>
#include <bitset>
#include <boost/aura/detail/svec.hpp>
#include <boost/aura/misc/sequence.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/misc/benchmark.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/device_map.hpp>
#include <boost/aura/copy.hpp>

using namespace boost::aura;
using namespace boost::aura::backend;

void bench_compute_only(device_array<float>& src,
                        device_array<float>& dst,
                        kernel& k,
                        feed& f)
{
	invoke(k, bounds(src.size()), args(dst.data(),
	                                   src.data()), f);
	wait_for(f);
}

void bench_compute_copy(std::vector<float>& hsrc,
                        std::vector<float>& hdst,
                        device_array<float>& src,
                        device_array<float>& dst,
                        kernel& k,
                        feed& f)
{
	boost::aura::copy(hsrc, src, f);
	invoke(k, bounds(src.size()), args(dst.data(),
	                                   src.data()), f);
	boost::aura::copy(dst, hdst, f);
	wait_for(f);
}

void bench_compute_map(std::vector<float>& hsrc,
                       std::vector<float>& hdst,
                       device_map<float>& src,
                       device_map<float>& dst,
                       kernel& k,
                       feed& f)
{
	src.remap(hsrc, f);
	dst.remap(hdst, f);
	invoke(k, bounds(src.size()), args(dst.data(),
	                                   src.data()), f);
	src.unmap(hsrc, f);
	dst.unmap(hdst, f);
	wait_for(f);
}

void bench_map_only(std::vector<float>& hsrc,
                    std::vector<float>& hdst,
                    device_map<float>& src,
                    device_map<float>& dst,
                    feed& f)
{
	src.remap(hsrc, f);
	dst.remap(hdst, f);
	src.unmap(hsrc, f);
	dst.unmap(hdst, f);
	wait_for(f);
}

void bench_compute_only_map(device_map<float>& src,
                            device_map<float>& dst,
                            kernel& k,
                            feed& f)
{
	invoke(k, bounds(src.size()), args(dst.data(),
	                                   src.data()), f);
	wait_for(f);
}

inline void run_tests(std::vector<svec<std::size_t, 1>> sizes,
                      int device_ordinal, std::size_t runtime)
{
	// benchmark result variables
	double min, max, mean, stdev;
	std::size_t runs;

	device d(device_ordinal);
	feed f(d);
	module m = create_module_from_file("zerocpy.cc", d,
	                                   AURA_BACKEND_COMPILE_FLAGS);
	kernel k = create_kernel(m, "compute");

	for (auto s : sizes) {
		std::vector<float> hsrc(s[0], 42.);
		std::vector<float> hdst(s[0], 0.);
		device_array<float> src(s[0], d);
		device_array<float> dst(s[0], d);

		copy(hsrc, src, f);
		bench_compute_only(src, dst, k, f);
		copy(dst, hdst, f);
		wait_for(f);
		if (!std::equal(hsrc.begin(), hsrc.end(), hdst.begin())) {
			std::cout << "compute_only FAILED" << std::endl;
		}
		AURA_BENCHMARK(bench_compute_only(src, dst, k, f),
		               runtime, min, max, mean, stdev, runs);
		print_benchmark_results("bench_compute_only",
		                        min, max, mean, stdev, runs, runtime);
	}

	for (auto s : sizes) {
		std::vector<float> hsrc(s[0], 42.);
		std::vector<float> hdst(s[0], 0.);
		device_array<float> src(s[0], d);
		device_array<float> dst(s[0], d);

		bench_compute_copy(hsrc, hdst, src, dst, k, f);
		if (!std::equal(hsrc.begin(), hsrc.end(), hdst.begin())) {
			std::cout << "compute_copy FAILED" << std::endl;
		}
		AURA_BENCHMARK(bench_compute_copy(hsrc, hdst, src, dst, k, f),
		               runtime, min, max, mean, stdev, runs);
		print_benchmark_results("compute_copy",
		                        min, max, mean, stdev, runs, runtime);
	}

	for (auto s : sizes) {
		std::vector<float> hsrc(s[0], 42.);
		std::vector<float> hdst(s[0], 0.);
		device_map<float> src(hsrc, memory_tag::ro, d);
		device_map<float> dst(hdst, memory_tag::wo, d);

		src.unmap(hsrc, f);
		dst.unmap(hdst, f);

		bench_compute_map(hsrc, hdst, src, dst, k, f);
		if (!std::equal(hsrc.begin(), hsrc.end(), hdst.begin())) {
			std::cout << "compute_map FAILED" << std::endl;
		}
		AURA_BENCHMARK(bench_compute_map(hsrc, hdst, src, dst, k, f),
		               runtime, min, max, mean, stdev, runs);
		print_benchmark_results("compute_map",
		                        min, max, mean, stdev, runs, runtime);
	}

	for (auto s : sizes) {
		std::vector<float> hsrc(s[0], 42.);
		std::vector<float> hdst(s[0], 0.);
		device_map<float> src(hsrc, memory_tag::ro, d);
		device_map<float> dst(hdst, memory_tag::wo, d);

		src.unmap(hsrc, f);
		dst.unmap(hdst, f);

		AURA_BENCHMARK(bench_map_only(hsrc, hdst, src, dst, f),
		               runtime, min, max, mean, stdev, runs);
		print_benchmark_results("map_only",
		                        min, max, mean, stdev, runs, runtime);
	}

	for (auto s : sizes) {
		std::vector<float> hsrc(s[0], 42.);
		std::vector<float> hdst(s[0], 0.);
		device_map<float> src(hsrc, memory_tag::ro, d);
		device_map<float> dst(hdst, memory_tag::wo, d);

		bench_compute_only_map(src, dst, k, f);
		src.unmap(hsrc, f);
		dst.unmap(hdst, f);
		wait_for(f);
		if (!std::equal(hsrc.begin(), hsrc.end(), hdst.begin())) {
			std::cout << "compute_only_map FAILED" << std::endl;
		}
		src.remap(hsrc, f);
		dst.remap(hdst, f);
		AURA_BENCHMARK(bench_compute_only_map(src, dst, k, f),
		               runtime, min, max, mean, stdev, runs);
		print_benchmark_results("compute_only_map",
		                        min, max, mean, stdev, runs, runtime);
	}
}


int main(int argc, char *argv[])
{

	initialize();

	// parse command line arguments:
	// -s memory size (sequence, max rank 3)
	// -d device (single value)
	// -t time (time per benchmark in ms)

	// config params
	std::vector<svec<std::size_t, 1> > sizes;
	int dev_ordinal = 0;
	std::size_t runtime = 1000;

	// parse config
	int opt;
	while ((opt = getopt(argc, argv, "s:d:t:")) != -1) {
		switch (opt) {
		case 's': {
			printf("size: %s ", optarg);
			sizes = boost::aura::generate_sequence<std::size_t, 1>(optarg);
			break;
		}
		case 'd': {
			printf("device %s ", optarg);
			dev_ordinal = atoi(optarg);
			break;
		}
		case 't': {
			runtime = atoi(optarg);
			printf("time: %lu ms ", runtime);
			// benchmark script expects us
			runtime *= 1000;
			break;
		}
		default: {
			fprintf(stderr, "Usage: %s -s <sizes> "
			        "-d <device ordinal> -t <runtime (ms)> <operations>\n", argv[0]);
			exit(-1);
		}
		}
	}

	// output info about selected device
	{
		device d(dev_ordinal);
		device_info di = device_get_info(d);
		printf("selected device: ");
		print_device_info(di);
	}
	run_tests(sizes, dev_ordinal, runtime);

}

