/**
 * compare the runtime of creating a std::vector and various alternatives
 * includes a variable number of copies of the object
 * result: constant size is fastest, dynamic is second, vector is slowest
 */

#include <vector>
#include <aura/misc/benchmark.hpp>

#define MAX_SIZE 11 

// -----

int test_vector_copy(const std::vector<int> & v) {
  int sum = 0;
  for(int i=0; i<(int)v.size(); i++) {
    sum += v[i];
  }
  return sum;
}

int test_vector(int size, int sums) {
  std::vector<int> v(size);
  for(int i=0; i<size; i++) {
    v[i] = i;
  }
  int sum = 0;
  for(int i=0; i<sums; i++) {
    sum += test_vector_copy(v);
  }
  return sum;
}

// -----

int test_const_size_copy(int v[MAX_SIZE], int size) {
  int sum = 0;
  for(int i=0; i<size; i++) {
    sum += v[i];
  }
  return sum;
}

int test_const_size(int size, int sums) {
  int v[MAX_SIZE]; 
  for(int i=0; i<size; i++) {
    v[i] = i;
  }
  int sum = 0;
  for(int i=0; i<sums; i++) {
    sum += test_const_size_copy(v, size);
  }
  return sum;
}

// -----

int test_dynamic_copy(int * v, int size) {
  int sum = 0;
  for(int i=0; i<size; i++) {
    sum += v[i];
  }
  return sum;
}

int test_dynamic(int size, int sums) {
  int * v = (int*)malloc(sizeof(int)*size); 
  for(int i=0; i<size; i++) {
    v[i] = i;
  }
  int sum = 0;
  for(int i=0; i<sums; i++) {
    sum += test_dynamic_copy(v, size);
  }
  free(v);
  return sum;

}

// -----

void run_test() {
  for(int i=1; i<MAX_SIZE; i++) {
    int duration = 100000; // 0.1s
    int copies = 3;
    // run benchmarks
    double min, max, mean, stdev;
    int num;
    int result = 0;
    MGPU_BENCHMARK(result += test_vector(i, copies), 
      duration, min, max, mean, stdev, num);
    printf("vector %d: [%1.2f %1.2f] %1.2f (%d) %d\n", 
      i, min, max, mean, num, result);
    result = 0;
    MGPU_BENCHMARK(result += test_const_size(i, copies), 
      duration, min, max, mean, stdev, num);
    printf("constant %d: [%1.2f %1.2f] %1.2f (%d) %d\n", 
      i, min, max, mean, num, result);
    result = 0;
    MGPU_BENCHMARK(result += test_dynamic(i, copies), 
      duration, min, max, mean, stdev, num);
    printf("dynamic %d: [%1.2f %1.2f] %1.2f (%d) %d\n\n", 
      i, min, max, mean, num, result);
  }
}


int main(void) {
  run_test();
}



