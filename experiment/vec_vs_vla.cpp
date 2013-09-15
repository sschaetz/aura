/**
 * compare the runtime of creating a std::vector and various alternatives
 * result: constant size if fastest, dynamic is second, vector is slowest
 */

#include <vector>
#include <aura/misc/benchmark.hpp>

#define MAX_SIZE 11 

// -----

int test_vector_copy(std::vector<int> v) {
  int sum = 0;
  for(int i=0; i<(int)v.size(); i++) {
    sum += v[i];
  }
  return sum;
}

int test_vector(int size) {
  std::vector<int> v(size);
  for(int i=0; i<size; i++) {
    v[i] = i;
  }
  return test_vector_copy(v);
}

// -----

int test_const_size_copy(int v[MAX_SIZE], int size) {
  int sum = 0;
  for(int i=0; i<size; i++) {
    sum += v[i];
  }
  return sum;
}

int test_const_size(int size) {
  int v[MAX_SIZE]; 
  for(int i=0; i<size; i++) {
    v[i] = i;
  }
  return test_const_size_copy(v, size);
}

// -----

int test_dynamic_copy(int * v, int size) {
  int sum = 0;
  for(int i=0; i<size; i++) {
    sum += v[i];
  }
  return sum;
}

int test_dynamic(int size) {
  int * v = (int*)malloc(sizeof(int)*size); 
  for(int i=0; i<size; i++) {
    v[i] = i;
  }
  int sum = test_dynamic_copy(v, size);
  free(v);
  return sum;
}

// -----

void run_test() {
  for(int i=1; i<MAX_SIZE; i++) {
    int duration = 100000; // 0.1s
    // run benchmarks
    double min, max, mean, stdev;
    int num;
    MGPU_BENCHMARK(test_vector(i), duration, min, max, mean, stdev, num);
    printf("vector %d: [%1.2f %1.2f] %1.2f (%d)\n", i, min, max, mean, num);
    MGPU_BENCHMARK(test_const_size(i), duration, min, max, mean, stdev, num);
    printf("constant %d: [%1.2f %1.2f] %1.2f (%d)\n", i, min, max, mean, num);
    MGPU_BENCHMARK(test_dynamic(i), duration, min, max, mean, stdev, num);
    printf("dynamic %d: [%1.2f %1.2f] %1.2f (%d)\n\n", i, min, max, mean, num);
  }
}


int main(void) {
  run_test();
}



