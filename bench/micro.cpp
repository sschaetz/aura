// run various micro-benchmarks (simple kernels) on all devices available

#include <vector>
#include <aura/backend.hpp>
#include <aura/misc/benchmark.hpp>

using namespace aura::backend;

#if AURA_BACKEND_OPENCL
const char * kernel_file = "bench/micro.cl"; 
#elif AURA_BACKEND_CUDA
const char * kernel_file = "bench/micro.ptx"; 
#endif

// run each subtest for a specific number of seconds
const int duration_per_test = 2*1e6;

// benchmark how long it takes to launch an empty kernel on 1...N GPUs

inline void bench_noarg_expr(std::vector<feed> & feeds, 
  std::vector<kernel> & kernels, std::size_t num) {
  for(std::size_t n=0; n<num; n++) {
    invoke(kernels[n], mesh(1), bundle(1), feeds[n]);
  } 
  for(std::size_t n=0; n<num; n++) {
    wait_for(feeds[n]); 
  } 
}


void bench_noarg(std::vector<device> & devices, 
  std::vector<feed> & feeds, const char * kernel_name) {
  
  double min, max, mean, stdev;
  std::size_t num;
  std::vector<module> modules(devices.size());
  std::vector<kernel> kernels(devices.size());
  for(std::size_t n=0; n<devices.size(); n++) {
    modules[n] = create_module_from_file(kernel_file, devices[n],
      AURA_BACKEND_COMPILE_FLAGS); 
    kernels[n] = create_kernel(modules[n], kernel_name);
    invoke(kernels[n], mesh(1), bundle(1), feeds[n]);
    wait_for(feeds[n]); 
  }
  for(std::size_t n=1; n<=devices.size(); n++) {
    AURA_BENCHMARK(bench_noarg_expr(feeds, kernels, n), 
      duration_per_test, min, max, mean, stdev, num);
    printf("%s_kernel: %ld GPUs num %lu min %f max %f mean %f stdev %f\n", 
      kernel_name, n, num, min, max, mean, stdev);
  }
}

// ----

// benchmark how long it takes to run a very simple kernel on 1...N GPUs

inline void bench_onearg_expr(std::vector<feed> & feeds, 
  std::vector<kernel> & kernels, std::vector<memory> & device_memory,
  std::size_t meshx, std::size_t meshy, std::size_t meshz, 
  std::size_t bundlex, std::size_t num) {
  for(std::size_t n=0; n<num; n++) {
    invoke(kernels[n], mesh(meshx, meshy, meshz), bundle(bundlex, 1, 1),
      args(device_memory[n]), feeds[n]);
  }
  for(std::size_t n=0; n<num; n++) {
    wait_for(feeds[n]); 
  } 
}


void bench_onearg(std::vector<device> & devices, 
  std::vector<feed> & feeds, const char * kernel_name, 
  std::size_t meshx, std::size_t meshy, std::size_t meshz, 
  std::size_t bundlex) {
  
  double min, max, mean, stdev;
  std::size_t num;
  std::size_t size = meshx * meshy * meshz;
  std::vector<module> modules(devices.size());
  std::vector<kernel> kernels(devices.size());
  std::vector<memory> device_memory(devices.size());
  std::vector<float> host_memory(size, 42.);
  for(std::size_t n=0; n<devices.size(); n++) {
    modules[n] = create_module_from_file(kernel_file, devices[n], 
      AURA_BACKEND_COMPILE_FLAGS); 
    kernels[n] = create_kernel(modules[n], kernel_name);
    device_memory[n] = device_malloc(size*sizeof(float), devices[n]);
    copy(&host_memory[0], device_memory[n], size*sizeof(float), feeds[n]);
    invoke(kernels[n], mesh(meshx, meshy, meshz), bundle(bundlex, 1, 1), 
      args(device_memory[n]), feeds[n]);
    wait_for(feeds[n]); 
  }

  for(std::size_t n=1; n<=devices.size(); n++) {
    AURA_BENCHMARK(bench_onearg_expr(feeds, kernels, 
        device_memory, meshx, meshy, meshz, bundlex, n), 
      duration_per_test, min, max, mean, stdev, num);
    printf("%s_kernel (%ldG %ld:%ld): num %lu min %f max %f mean %f stdev %f\n",
      kernel_name, n, meshx*meshy*meshz, bundlex, num, min, max, mean, stdev);
  }
}

// ----

int main(void) {
  initialize();
  print_device_info();
  std::size_t num = 1;

  std::vector<device> devices;
  devices.reserve(num);
  std::vector<feed> feeds;
  feeds.reserve(num);
  for(std::size_t n=0; n<num; n++) {
    devices.push_back(device(n));
    feeds.push_back(feed(devices[n]));   
  }

  bench_noarg(devices, feeds, "noarg");
  bench_onearg(devices, feeds, "simple_add", 32, 32, 32, 32);
  bench_onearg(devices, feeds, "four_mad", 32, 32, 32, 32);
  bench_onearg(devices, feeds, "peak_flop_empty", 256, 256, 256, 256);
  bench_onearg(devices, feeds, "peak_flop", 256, 256, 256, 256);
}


