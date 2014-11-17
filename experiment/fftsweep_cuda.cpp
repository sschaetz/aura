
#include <complex>
#include <cufft.h>
#include <boost/aura/backend.hpp>
#include <boost/aura/misc/benchmark.hpp>

using namespace boost::aura::backend;

/// benchmark 2d fft performance

// number of ffts in parallel
#define batch_size 3
// first fft size
#define start_size 256
// last fft size
#define end_size 1025 
// step size between sizes
#define step_size 1
// runtime per test in seconds
#define runtime 2 

typedef std::complex<float> cfloat;

void run_test(int size, device & d, feed & f) {
  // allocate memory 
  device_ptr<cfloat> m1 = device_malloc<cfloat>(size*size*batch_size, d);
  device_ptr<cfloat> m2 = device_malloc<cfloat>(size*size*batch_size, d);
 
  // allocate fft handle
  cufftHandle plan;
  int dims[2] = { size, size };
  int embed[2] = { size * size, size };
  AURA_CUFFT_SAFE_CALL(cufftPlanMany(&plan, 2, dims, embed, 1, size * size, 
    embed, 1, size * size, CUFFT_C2C, batch_size));
  
  // run test fft (warmup)
  AURA_CUFFT_SAFE_CALL(cufftExecC2C(plan, (cufftComplex *)m1.get(),
    (cufftComplex *)m2.get(), CUFFT_FORWARD));

  // run benchmark
  double min, max, mean, stdev;
  int num;
  AURA_BENCHMARK_ASYNC(cufftExecC2C(plan, (cufftComplex *)m1.get(),
    (cufftComplex *)m2.get(), CUFFT_FORWARD), f.synchronize();, 
    runtime, min, max, mean, stdev, num);
 
  // print result
  printf("%d: [%1.2f %1.2f] %1.2f %1.2f %d\n", 
    size, min, max, mean, stdev, num);
  f.synchronize();
  AURA_CUFFT_SAFE_CALL(cufftDestroy(plan));
  
  device_free(m1);
  device_free(m2);
}


int main(void) {
  initialize();
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
        d1.unpin();
        d0.pin();
      }
      run_test(i, d0, f0);
    } else {
      if(1 != first) {
        first = 1;
        d0.unpin();
        d1.pin();
      }
      run_test(i, d1, f1);
    }
  }
}



