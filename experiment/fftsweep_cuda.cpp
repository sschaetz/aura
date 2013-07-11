
#include <cufft.h>
#include <aura/backend.hpp>
#include <aura/misc/benchmark.hpp>

using namespace aura::backend;

/// benchmark 2d fft performance

// number of ffts in parallel
#define batch_size 3
// first fft size
#define start_size 270 
// last fft size
#define end_size 1024 
// step size between sizes
#define step_size 1
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
  device d0(0);
  device d1(4);
  feed f0(d0);
  feed f1(d1);

  int first = -1;
  for(int i=start_size; i<=end_size; i=i+step_size) {
    if((i/10) % 2)
    {
      if(0 != first) {
        first = 0;
        f1.unpin();
        f0.pin();
      }
      run_test(i, f0);
    } else {
      if(1 != first) {
        first = 1;
        f0.unpin();
        f1.pin();
      }
      run_test(i, f1);
    }
  }
}



