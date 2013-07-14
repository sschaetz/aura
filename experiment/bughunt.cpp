// run 2D ffts on all GPUs in the system continuously
// designed to stress-test a (multi-)GPU system

#include <vector>
#include <cufft.h>
#include <aura/backend.hpp>
#include <aura/misc/benchmark.hpp>

using namespace aura::backend;

/// benchmark 2d fft performance

// number of ffts in parallel
#define batch_size 3

// runtime per test in seconds
#define runtime 10 


void run_test(int size, feed & f) {
  // allocate memory
  memory m1 = device_malloc(size*size*sizeof(float)*2*batch_size, f);
  memory m2 = device_malloc(size*size*sizeof(float)*2*batch_size, f);
 
  // allocate fft handle
  cufftHandle plan;
  int dims[2] = { size, size };
  int embed[2] = { size * size, size };
  AURA_CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, dims, embed, 1, size * size, 
    embed, 1, size * size, CUFFT_C2C, batch_size));
  
  // run test fft (warmup)
  AURA_CUFFT_SAFE_CALL(cufftExecC2C(plan, (cufftComplex *)m1,
    (cufftComplex *)m2, CUFFT_FORWARD));

  // run benchmark
  double min, max, mean, stdev;
  int num;
  MGPU_BENCHMARK_ASYNC(cufftExecC2C(plan, (cufftComplex *)m1,
    (cufftComplex *)m2, CUFFT_FORWARD), ;, size/runtime/10, 
    min, max, mean, stdev, num);
 
  // print result
  printf("%d: [%1.2f %1.2f] %1.2f %1.2f %d\n", 
    size, min, max, mean, stdev, num);
  f.synchronize();
  AURA_CUFFT_SAFE_CALL(cufftDestroy(plan));
  
  device_free(m1, f);
  device_free(m2, f);
}


int main(void) {
  init();
  int num = device_get_count();
  if(1 >= num) {
    printf("no devices found\n"); exit(0);
  }

  std::vector<device *> devices;
  std::vector<feed *> feeds;
  for(int i=0; i<num; i++) {
    devices.push_back(new device(i)); 
    feeds.push_back(new feed(*devices[devices.size()-1])); 
  }

  int dev = 0;
  while(true) {
    feeds[dev]->pin();
    run_test(486, *feeds[dev]);
    feeds[dev]->unpin();
    dev = (dev+1)%8;
  }
}



