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


void run_test(int size, device & d, feed & f) {
  // allocate memory
  memory m1 = device_malloc(size*size*sizeof(float)*2*batch_size, d);
  memory m2 = device_malloc(size*size*sizeof(float)*2*batch_size, d);
 
  f.set(); 
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
  f.set(); 
  AURA_CUFFT_SAFE_CALL(cufftDestroy(plan));
  device_free(m1, d);
  device_free(m2, d);
}


int main(void) {
  init();
  int num = device_get_count();
  if(1 >= num) {
    printf("no devices found\n"); exit(0);
  }
  int size = 486;
  std::vector<device *> devices;
  std::vector<feed *> feeds;
  std::vector<cufftHandle> plans;
  std::vector<memory> memories1;
  std::vector<memory> memories2;
  for(int i=0; i<num; i++) {
    devices.push_back(new device(i)); 
    cufftHandle plan;
    int dims[2] = { size, size };
    int embed[2] = { size * size, size };
    devices[devices.size()-1]->set();
    AURA_CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, dims, embed, 1, size * size, 
      embed, 1, size * size, CUFFT_C2C, batch_size));
    devices[devices.size()-1]->unset();
    plans.push_back(plan); 
    feeds.push_back(new feed(*devices[devices.size()-1])); 
    memory m1 = device_malloc(size*size*sizeof(float)*2*batch_size, 
      *devices[devices.size()-1]);
    memory m2 = device_malloc(size*size*sizeof(float)*2*batch_size, 
      *devices[devices.size()-1]);
    memories1.push_back(m1);
    memories2.push_back(m2);
  }

  int dev = 0;
  int it = 0;
  while(true) {
    devices[dev]->set(); 
    AURA_CUFFT_SAFE_CALL(cufftExecC2C(plans[dev], 
      (cufftComplex *)memories1[dev],
      (cufftComplex *)memories2[dev], CUFFT_FORWARD));
    dev = (dev+1)%num; 
    it++;
    // synchronize every 3 calls
    if((it-dev)%3==0) {
      AURA_CUDA_SAFE_CALL(cuCtxSynchronize()); 
    }
    printf("%d\r", it);
  }

}



