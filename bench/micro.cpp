#include <vector>
#include <aura/backend.hpp>
#include <aura/misc/benchmark.hpp>

using namespace aura::backend;

#if AURA_BACKEND_OPENCL
const char * kernel_file = "bench/micro.cl"; 
#elif AURA_BACKEND_CUDA
const char * kernel_file = "bench/micro.ptx"; 
#endif


inline void bench_noarg_expr(std::vector<feed *> & feeds, 
  std::vector<kernel> & kernels) {
  for(std::size_t n=0; n<feeds.size(); n++) {
    invoke(kernels[n], grid(1), block(1), *feeds[n]);
  } 
  for(std::size_t n=0; n<feeds.size(); n++) {
    wait_for(*feeds[n]); 
  } 
}


void bench_noarg(std::vector<device *> & devices, 
  std::vector<feed *> & feeds) {
  
  double min, max, mean, stdev;
  std::size_t num;
  std::vector<module> modules(devices.size());
  std::vector<kernel> kernels(devices.size());
  for(std::size_t n=0; n<devices.size(); n++) {
    modules[n] = create_module_from_file(kernel_file, *devices[n]); 
    kernels[n] = create_kernel(modules[n], "noarg");
    invoke(kernels[n], grid(1), block(1), *feeds[n]);
  }

  MGPU_BENCHMARK(bench_noarg_expr(feeds, kernels), 
    4, min, max, mean, stdev, num);
  printf("noarg: num %lu min %f max %f mean %f stdev %f\n", 
    num, min, max, mean, stdev);
  MGPU_BENCHMARK_HISTOGRAM(bench_noarg_expr(feeds, kernels), 
    4, min);
}


int main(void) {
  init();
  std::size_t num = device_get_count();

  std::vector<device *> devices(num);
  std::vector<feed *> feeds(num);
  for(std::size_t n=0; n<num; n++) {
    devices[n] = new device(n);   
    feeds[n] = new feed(*devices[n]);   
  }

  bench_noarg(devices, feeds);
}


