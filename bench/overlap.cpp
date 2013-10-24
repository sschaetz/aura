// idea of benchmark: try to see if we can overlap a p2p kernel and FFT 
//
// use 4 devices
//
// in the same feed:
// run 2D FFTs (variable batchsize) on all
// run a all to all copy kernel
//
// try the same with different feeds
// vary the batch size to see if overlap performance changes
// 
// result: apparently it does not work as expected, almost no overlaping can
// be observed
//


#include <complex>
#include <vector>
#include <algorithm>
#include <aura/backend.hpp>
#include <aura/misc/benchmark.hpp>

using namespace aura::backend;

#if AURA_BACKEND_OPENCL
const char * kernel_file = "bench/overlap.cl"; 
#elif AURA_BACKEND_CUDA
const char * kernel_file = "bench/overlap.ptx"; 
#endif

typedef std::complex<float> cfloat;

// run each subtest for a specific number of seconds
const int duration_per_test = 2*1e6;

void bench_fft_only(std::vector<memory> & fftmem1, 
    std::vector<memory> & fftmem2, 
    std::vector<memory> & p2pmem1, 
    std::vector<memory> & p2pmem2, 
    std::vector<fft> & ffth, 
    std::vector<kernel> & kernels, 
    std::vector<feed> & feeds1,
    std::vector<feed> & feeds2,
    std::size_t dim) {
  for(std::size_t n = 0; n<feeds1.size(); n++) {
    fft_forward(fftmem1[n], fftmem2[n], ffth[n], feeds1[n]);
  }
  std::for_each(feeds1.begin(), feeds1.end(), &wait_for);
}

void bench_overlap_same_feed(std::vector<memory> & fftmem1, 
    std::vector<memory> & fftmem2, 
    std::vector<memory> & p2pmem1, 
    std::vector<memory> & p2pmem2, 
    std::vector<fft> & ffth, 
    std::vector<kernel> & kernels, 
    std::vector<feed> & feeds1,
    std::vector<feed> & feeds2,
    std::size_t dim) {
  for(std::size_t n = 0; n<feeds1.size(); n++) {
    invoke(kernels[n], grid(dim/2), block(dim/2), 
      args(p2pmem1[n], 
        p2pmem2[(n+1)%feeds1.size()],
        p2pmem2[(n+2)%feeds1.size()],
        p2pmem2[(n+3)%feeds1.size()]
      ), feeds1[n]);
    fft_forward(fftmem1[n], fftmem2[n], ffth[n], feeds1[n]);
  }
  std::for_each(feeds1.begin(), feeds1.end(), &wait_for);
}

void bench_overlap_diff_feed(std::vector<memory> & fftmem1, 
    std::vector<memory> & fftmem2, 
    std::vector<memory> & p2pmem1, 
    std::vector<memory> & p2pmem2, 
    std::vector<fft> & ffth, 
    std::vector<kernel> & kernels, 
    std::vector<feed> & feeds1,
    std::vector<feed> & feeds2,
    std::size_t dim) {
  for(std::size_t n = 0; n<feeds1.size(); n++) {
    invoke(kernels[n], grid(dim/2), block(dim/2), 
      args(p2pmem1[n], 
        p2pmem2[(n+1)%feeds1.size()],
        p2pmem2[(n+2)%feeds1.size()],
        p2pmem2[(n+3)%feeds1.size()]
      ), feeds1[n]);
    fft_forward(fftmem1[n], fftmem2[n], ffth[n], feeds2[n]);
  }
  std::for_each(feeds2.begin(), feeds2.end(), &wait_for);
  std::for_each(feeds1.begin(), feeds1.end(), &wait_for);
}

void bench_overlap_diff_feed_2(std::vector<memory> & fftmem1, 
    std::vector<memory> & fftmem2, 
    std::vector<memory> & p2pmem1, 
    std::vector<memory> & p2pmem2, 
    std::vector<fft> & ffth, 
    std::vector<kernel> & kernels, 
    std::vector<feed> & feeds1,
    std::vector<feed> & feeds2,
    std::size_t dim) {
  
  fft_forward(fftmem1[0], fftmem2[0], ffth[0], feeds2[0]);
  fft_forward(fftmem1[1], fftmem2[1], ffth[1], feeds2[1]);
  fft_forward(fftmem1[2], fftmem2[2], ffth[2], feeds2[2]);
  fft_forward(fftmem1[3], fftmem2[3], ffth[3], feeds2[3]);
  
  // 0 + 1
  invoke(kernels[0], grid(dim/2), block(dim/2), 
    args(p2pmem1[0], p2pmem2[0], p2pmem2[0+1]), feeds1[0]);
  // 2+3
  invoke(kernels[2], grid(dim/2), block(dim/2), 
    args(p2pmem1[2], p2pmem2[2], p2pmem2[2+1]), feeds1[2]);
  std::for_each(feeds1.begin(), feeds1.end(), &wait_for);
  // 01 + 23 
  invoke(kernels[0], grid(dim/2), block(dim/2), 
    args(p2pmem2[0], p2pmem1[0], p2pmem1[2]), feeds1[0]);
  // 23 + 01 
  invoke(kernels[2], grid(dim/2), block(dim/2), 
    args(p2pmem2[2], p2pmem1[2], p2pmem1[0]), feeds1[2]);
  std::for_each(feeds1.begin(), feeds1.end(), &wait_for);
  invoke(kernels[1], grid(dim/2), block(dim/2), 
    args(p2pmem1[1], p2pmem1[1], p2pmem2[0]), feeds1[1]);
  invoke(kernels[3], grid(dim/2), block(dim/2), 
    args(p2pmem1[3], p2pmem1[3], p2pmem2[2]), feeds1[3]);
  std::for_each(feeds1.begin(), feeds1.end(), &wait_for);
  
  std::for_each(feeds2.begin(), feeds2.end(), &wait_for);
}

void bench_overlap(std::vector<device> & devices, 
  std::vector<feed> & feeds1, std::vector<feed> & feeds2,  
  std::size_t dim, std::size_t batch) {

  // benchmark result variables
  double min, max, mean, stdev;
  std::size_t num;
  
  // allocate memory for fft and p2p test
  std::vector<memory> fftmem1(devices.size()); 
  std::vector<memory> fftmem2(devices.size()); 
  std::vector<memory> p2pmem1(devices.size());
  std::vector<memory> p2pmem2(devices.size());
  for(std::size_t n=0; n<devices.size(); n++) {
    fftmem1[n] = device_malloc(batch*dim*dim*sizeof(cfloat), devices[n]);
    fftmem2[n] = device_malloc(batch*dim*dim*sizeof(cfloat), devices[n]);
    p2pmem1[n] = device_malloc(dim*dim*sizeof(cfloat), devices[n]);
    p2pmem2[n] = device_malloc(dim*dim*sizeof(cfloat), devices[n]);
  }

  // create fft handle
  std::vector<fft> ffth(devices.size());
  for(std::size_t n=0; n<devices.size(); n++) {
    ffth[n] = fft(devices[n], fft_dim(dim, dim), fft::type::c2c, batch);
  }

  // create kernel handle
  std::vector<module> modules(devices.size());
  std::vector<kernel> kernels_2(devices.size());
  std::vector<kernel> kernels_4(devices.size());
  for(std::size_t n=0; n<devices.size(); n++) {
    modules[n] = create_module_from_file(kernel_file, devices[n]); 
    kernels_2[n] = create_kernel(modules[n], "p2p_2");
    kernels_4[n] = create_kernel(modules[n], "p2p_4");
  }

  // enable p2p communication
  enable_peer_access(devices[0], devices[1]);    
  enable_peer_access(devices[0], devices[2]);    
  enable_peer_access(devices[0], devices[3]);    
  enable_peer_access(devices[1], devices[2]);    
  enable_peer_access(devices[1], devices[3]);    
  enable_peer_access(devices[2], devices[3]);    

  // warmup run 
  bench_overlap_same_feed(fftmem1, fftmem2, p2pmem1, p2pmem2, 
    ffth, kernels_4, feeds1, feeds2, dim);
  // synchronize
  std::for_each(feeds1.begin(), feeds1.end(), &wait_for);
  
  MGPU_BENCHMARK(bench_fft_only(fftmem1, fftmem2, p2pmem1, p2pmem1, 
      ffth, kernels_4, feeds1, feeds2, dim),
    duration_per_test, min, max, mean, stdev, num);
  printf("%s: GPUs num %lu min %f max %f mean %f stdev %f\n", 
    "bench_fft_only", num, min, max, mean, stdev);
  
  MGPU_BENCHMARK(bench_overlap_same_feed(fftmem1, fftmem2, p2pmem1, p2pmem1, 
      ffth, kernels_4, feeds1, feeds2, dim),
    duration_per_test, min, max, mean, stdev, num);
  printf("%s: GPUs num %lu min %f max %f mean %f stdev %f\n", 
    "bench_overlap_same_feed", num, min, max, mean, stdev);
  
  MGPU_BENCHMARK(bench_overlap_diff_feed(fftmem1, fftmem2, p2pmem1, p2pmem1, 
      ffth, kernels_4, feeds1, feeds2, dim),
    duration_per_test, min, max, mean, stdev, num);
  printf("%s: GPUs num %lu min %f max %f mean %f stdev %f\n", 
    "bench_overlap_diff_feed", num, min, max, mean, stdev);
 
  MGPU_BENCHMARK(bench_overlap_diff_feed_2(fftmem1, fftmem2, p2pmem1, p2pmem1, 
      ffth, kernels_2, feeds1, feeds2, dim),
    duration_per_test, min, max, mean, stdev, num);
  printf("%s: GPUs num %lu min %f max %f mean %f stdev %f\n", 
    "bench_overlap_diff_feed_2", num, min, max, mean, stdev);
  
  for(std::size_t n=0; n<devices.size(); n++) {
    device_free(fftmem1[n], devices[n]);
    device_free(fftmem2[n], devices[n]);
    device_free(p2pmem1[n], devices[n]);
    device_free(p2pmem2[n], devices[n]);
  }

}

int main(void) {
  initialize();
  std::size_t num = device_get_count();
  if(num < 4) {
    printf("not enough devices for overlap benchmark!\n");  
  }

  // work with 4 devices
  num = 4;

  std::vector<device> devices;
  devices.reserve(num);
  std::vector<feed> feeds1;
  std::vector<feed> feeds2;
  feeds1.reserve(num);
  feeds2.reserve(num);
  for(std::size_t n=0; n<num; n++) {
    devices.push_back(device(n));
    feeds1.push_back(feed(devices[n]));   
    feeds2.push_back(feed(devices[n]));   
  }

  bench_overlap(devices, feeds1, feeds2, 512, 3);
}

