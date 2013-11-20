// run peak benchmarks:
//
// * single performance with multiply-add kernel
// * double performance with multiply-add kernel
// * ondevice throughput (copy)
// * ondevice throughput (scale)
// * ondevice throughput (sum) 
// * ondevice throughput (triad) 
// * bus throughput (host to device) 
// * bus throughput (device to host) 


#include <iostream>
#include <bitset>
#include <algorithm>
#include <vector>
#include <aura/backend.hpp>
#include <aura/misc/sequence.hpp>
#include <aura/misc/benchmark.hpp>

const char * ops_tbl[] = { "sflop", "dflop", "devcopy", "devscale", 
  "devsum", "devtriad", "tphtd", "tpdth" };

using namespace aura;
using namespace aura::backend;

#if AURA_BACKEND_OPENCL
const char * kernel_file = "bench/peak.cl"; 
#elif AURA_BACKEND_CUDA
const char * kernel_file = "bench/peak.ptx"; 
#endif

inline void run_host_to_device(feed & f, memory dst, 
  std::vector<float> & src) {
  copy(dst, &src[0], src.size()*sizeof(float), f); 
  wait_for(f);
}

inline void run_device_to_host(feed & f, std::vector<float> & dst, 
  memory src) {
  copy(&dst[0], src, dst.size()*sizeof(float), f); 
  wait_for(f);
}

inline void run_tests(
  std::vector<svec<std::size_t, AURA_MAX_MESH_DIMS> > & meshes, 
  std::vector<svec<std::size_t, AURA_MAX_BUNDLE_DIMS> > & bundles,
  std::vector<svec<std::size_t, 1> > & sizes,
  std::vector<svec<std::size_t, 1> > & dev_ordinals, std::size_t runtime,
  std::bitset< sizeof(ops_tbl)/sizeof(ops_tbl[0]) > & ops) {

  // benchmark result variables
  double min, max, mean, stdev;
  std::size_t runs;
  
  device d(dev_ordinals[0][0]);
  feed f(d);

  if(!ops[6] && !ops[7]) {
    return;
  }

  for(std::size_t s=0; s<sizes.size(); s++) {
    std::vector<float> a1(sizes[s][0], 42.);
    std::vector<float> a2(sizes[s][0]);
    memory m = device_malloc(sizes[s][0]*sizeof(float), d);
    
    if(ops[6]) { // tphtd
      run_host_to_device(f, m, a1);
      copy(&a2[0], m, a2.size()*sizeof(float), f);
      wait_for(f);
      
      if(!std::equal(a1.begin(), a1.end(), a2.begin())) {
        printf("%s failed!\n", ops_tbl[6]);
        return;
      } 
      
      AURA_BENCHMARK(run_host_to_device(f, m, a1), runtime, min, max, 
        mean, stdev, runs);
      std::cout << ops_tbl[6] << " (" << sizes[s] << ") min " << min << 
        " max " << max << " mean " << mean << " stdev " << stdev << 
        " runs " << runs << " runtime " << runtime << std::endl;
    }
    
    if(ops[7]) { // tpdth
      std::fill(a2.begin(), a2.end(), 0.0);
      copy(m, &a1[0], a1.size()*sizeof(float), f);
      run_device_to_host(f, a2, m);
      
      if(!std::equal(a1.begin(), a1.end(), a2.begin())) {
        printf("%s failed!\n", ops_tbl[7]);
        return;
      } 
      
      AURA_BENCHMARK(run_device_to_host(f, a2, m), runtime, min, max,
        mean, stdev, runs);
      std::cout << ops_tbl[6] << " (" << sizes[s] << ") min " << min << 
        " max " << max << " mean " << mean << " stdev " << stdev << 
        " runs " << runs << " runtime " << runtime << std::endl;
    }
  }
}


int main(int argc, char *argv[]) {

  initialize();
  
  // parse command line arguments:
  // -m mesh sizes (sequence, max rank 3)
  // -b bundle sizes (sequence, max rank 3)
  // -s size of vector used for test (sequence, max rank 1)
  // -d device (single value or pair for device to device)
  // -t time (time per benchmark in ms)

  // config params
  std::bitset< sizeof(ops_tbl)/sizeof(ops_tbl[0]) > ops;
  std::vector<svec<std::size_t, AURA_MAX_MESH_DIMS> > meshes;
  std::vector<svec<std::size_t, AURA_MAX_BUNDLE_DIMS> > bundles;
  std::vector<svec<std::size_t, 1> > sizes;
  std::vector<svec<std::size_t, 1> > dev_ordinals;
  std::size_t runtime = 0;
 
  // parse config
  int opt;
  while ((opt = getopt(argc, argv, "m:b:s:d:t:")) != -1) {
    switch (opt) {
      case 'm': {
        printf("mesh: %s ", optarg);
        meshes = aura::generate_sequence<std::size_t, 
               AURA_MAX_MESH_DIMS>(optarg);
        break;
      }
      case 'b': {
        printf("bundle: %s ", optarg);
        bundles = aura::generate_sequence<std::size_t, 
                AURA_MAX_BUNDLE_DIMS>(optarg);
        break;
      }
      case 's': {
        printf("size: %s ", optarg);
        sizes = aura::generate_sequence<std::size_t, 1>(optarg);
        break;
      }
      case 'd': {
        printf("device %s ", optarg);
        dev_ordinals = aura::generate_sequence<std::size_t, 1> (optarg);
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
        fprintf(stderr, "Usage: %s -m <meshsize> -b <bundlesize> "
          "-s <vectorsize> -d <device ordinal (1 or 2)> -t <runtime (ms)> "
          "<operations>\n", argv[0]);
        fprintf(stderr, "Operations are: ");
        for(unsigned int i=0; i<sizeof(ops_tbl)/sizeof(ops_tbl[0]); i++) {
          fprintf(stderr, "%s ", ops_tbl[i]);
        }
        fprintf(stderr, "\n");
        exit(-1);
      }
    }
  }
  printf("operations: ");
  for(unsigned int i=0; i<sizeof(ops_tbl)/sizeof(ops_tbl[0]); i++) {
    ops[i] = false;
    for(int j=optind; j<argc; j++) {
      if(NULL != strstr(argv[j], ops_tbl[i])) {
        printf("%s ", ops_tbl[i]);
        ops[i] = true;
      }
    }
  }
  printf("\n");
  
  // output info about selected device  
  {
    printf("selected device(s): ");
    for(std::size_t i=0; i<dev_ordinals.size(); i++) {
      device d(dev_ordinals[i][0]);
      device_info di = device_get_info(d);
      print_device_info(di); 
    }
  }

  run_tests(meshes, bundles, sizes, dev_ordinals, runtime, ops);

}
