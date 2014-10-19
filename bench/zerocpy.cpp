// run copy and zero copy benchmarks:
//
// * device copy (to and from device)
// * map memory and copy using kernel

#include <vector>
#include <tuple>
#include <bitset>
#include <aura/detail/svec.hpp>
#include <aura/misc/sequence.hpp>
#include <aura/backend.hpp>
#include <aura/misc/benchmark.hpp>
#include <aura/device_array.hpp>
#include <aura/device_view.hpp>
#include <aura/copy.hpp>

using namespace aura;
using namespace aura::backend;

void bench_copy_up_down(std::vector<float>& src,
		device_array<float>& tmp,
		std::vector<float>& dst,
		feed& f)
{
	aura::copy(tmp, src, f);
	aura::copy(dst, tmp, f);
	wait_for(f);
}

void bench_map_up_down(std::vector<float>& src,
		std::vector<float>& dst,
		kernel& k,
		feed& f,
		device& d)
{
	device_view<float> srcv = map<float>(src, d);
	device_view<float> dstv = map<float>(dst, d);
	invoke(k, bounds(srcv.size()), args(dstv.begin_ptr(), 
				srcv.begin_ptr()), f);
	unmap(srcv, src, f);
	unmap(dstv, dst, f);
	wait_for(f);
}

void bench_map_only(std::vector<float>& src,
		std::vector<float>& dst,
		feed& f,
		device& d)
{
	device_view<float> srcv = map<float>(src, d);
	device_view<float> dstv = map<float>(dst, d);
	unmap(srcv, src, f);
	unmap(dstv, dst, f);
	wait_for(f);
}

void bench_kernel_copy_only(std::vector<float>& src,
		std::vector<float>& dst,
		feed& f,
		device& d)
{
	device_view<float> srcv = map<float>(src, d);
	device_view<float> dstv = map<float>(dst, d);
	unmap(srcv, src, f);
	unmap(dstv, dst, f);
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
	
	for (auto s : sizes) {

		std::vector<float> src(s[0], 42.);
		std::vector<float> dst(s[0], 1.);
		device_array<float> tmp(s[0], d);

		// bench_copy_up_down test
		bench_copy_up_down(src, tmp, dst, f);
		if (!std::equal(src.begin(), src.end(), dst.begin())) {
			std::cout << "bench_copy_up_down FAILED" << std::endl;
		}
		AURA_BENCHMARK(bench_copy_up_down(src, tmp, dst, f), 
				runtime, min, max, mean, stdev, runs);
		print_benchmark_results("bench_copy_up_down", 
				min, max, mean, stdev, runs, runtime);
	}


	module m = create_module_from_file("zerocpy.cc", d, 
			AURA_BACKEND_COMPILE_FLAGS);
	kernel k = create_kernel(m, "copy");
	for (auto s : sizes) {

		std::vector<float> src(s[0], 42.);
		std::vector<float> dst(s[0], 1.);

		// bench_copy_up_down test
		bench_map_up_down(src, dst, k, f, d);
		if (!std::equal(src.begin(), src.end(), dst.begin())) {
			std::cout << "bench_map_up_down FAILED" << std::endl;
		}
		AURA_BENCHMARK(bench_map_up_down(src, dst, k, f, d), 
				runtime, min, max, mean, stdev, runs);
		print_benchmark_results("bench_map_up_down", 
				min, max, mean, stdev, runs, runtime);
	}

	for (auto s : sizes) {

		std::vector<float> src(s[0], 42.);
		std::vector<float> dst(s[0], 1.);

		// bench_copy_up_down test
		bench_map_only(src, dst, f, d);
		AURA_BENCHMARK(bench_map_only(src, dst, f, d), 
				runtime, min, max, mean, stdev, runs);
		print_benchmark_results("bench_map_only", 
				min, max, mean, stdev, runs, runtime);
	}
}


int main(int argc, char *argv[]) {

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
        sizes = aura::generate_sequence<std::size_t, 1>(optarg);
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

