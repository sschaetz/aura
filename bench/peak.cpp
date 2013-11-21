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
  "devadd", "devtriad", "tphtd", "tpdth" };

using namespace aura;
using namespace aura::backend;

const char * kernel_file = "bench/peak.cc"; 

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

inline void run_kernel(feed & f, kernel & k, 
  mesh & m, bundle & b, memory m1) {
  invoke(k, m, b, args(m1), f);
  wait_for(f); 
}

inline void run_kernel(feed & f, kernel & k, 
  mesh & m, bundle & b, memory m1, memory m2) {
  invoke(k, m, b, args(m1, m2), f);
  wait_for(f); 
}

inline void run_kernel_f(feed & f, kernel & k, 
  mesh & m, bundle & b, memory m1, memory m2, float s) {
  invoke(k, m, b, args(m1, m2, s), f);
  wait_for(f); 
}

inline void run_kernel(feed & f, kernel & k, 
  mesh & m, bundle & b, memory m1, memory m2, memory m3) {
  invoke(k, m, b, args(m1, m2, m3), f);
  wait_for(f); 
}

inline void run_kernel_f(feed & f, kernel & k, 
  mesh & m, bundle & b, memory m1, memory m2, memory m3, float s) {
  invoke(k, m, b, args(m1, m2, m3, s), f);
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

  module m = create_module_from_file(kernel_file, d, 
    AURA_BACKEND_COMPILE_FLAGS);
 
  if(ops[0] || ops[1]) {
    kernel ksflop = create_kernel(m, "peak_flop_single"); 
    kernel kdflop = create_kernel(m, "peak_flop_double"); 
    for(std::size_t m=0; m<meshes.size(); m++) {
      std::size_t vsize = product(meshes[m]);
      memory mems = device_malloc(vsize*sizeof(float), d);
      memory memd = device_malloc(vsize*sizeof(double), d);
      for(std::size_t b=0; b<bundles.size(); b++) {
        if(ops[0]) { // sflop
          run_kernel(f, ksflop, meshes[m], bundles[b], mems);
          AURA_BENCHMARK(run_kernel(f, ksflop, meshes[m], 
            bundles[b], mems), runtime, min, max, mean, stdev, runs);
          std::cout << ops_tbl[0] << " (" << vsize << ") mesh (" << 
            meshes[m] << ") bundle (" << bundles[b] << ") min " << min << 
            " max " << max << " mean " << mean << " stdev " << stdev << 
            " runs " << runs << " runtime " << runtime << std::endl;
        }
        if(ops[1]) { // dflop
          run_kernel(f, kdflop, meshes[m], bundles[b], memd);
          AURA_BENCHMARK(run_kernel(f, kdflop, meshes[m], 
            bundles[b], memd), runtime, min, max, mean, stdev, runs);
          std::cout << ops_tbl[1] << " (" << vsize << ") mesh (" <<
            meshes[m] << ") bundle (" << bundles[b] << ") min " << min <<
            " max " << max << " mean " << mean << " stdev " << stdev <<
            " runs " << runs << " runtime " << runtime << std::endl;
        }
      }
      device_free(mems, d);
      device_free(memd, d);
    }
  }

  if(ops[2] || ops[3] || ops[4] || ops[5]) {
    kernel kcopy = create_kernel(m, "peak_copy"); 
    kernel kscale = create_kernel(m, "peak_scale"); 
    kernel kadd = create_kernel(m, "peak_add"); 
    kernel ktriad = create_kernel(m, "peak_triad"); 
    for(std::size_t m=0; m<meshes.size(); m++) {
      std::size_t vsize = product(meshes[m])*64;
      memory mem1 = device_malloc(vsize*sizeof(float), d);
      memory mem2 = device_malloc(vsize*sizeof(float), d);
      memory mem3 = device_malloc(vsize*sizeof(float), d);
      for(std::size_t b=0; b<bundles.size(); b++) {
        if(ops[2]) { // copy 
          run_kernel(f, kcopy, meshes[m], bundles[b], mem1, mem2);
          AURA_BENCHMARK(run_kernel(f, kcopy, meshes[m], bundles[b], 
            mem1, mem2), runtime, min, max, mean, stdev, runs);
          std::cout << ops_tbl[2] << " (" << vsize << ") mesh (" << 
            meshes[m] << ") bundle (" << bundles[b] << ") min " << min << 
            " max " << max << " mean " << mean << " stdev " << stdev << 
            " runs " << runs << " runtime " << runtime << std::endl;
        }
        if(ops[3]) { // scale 
          run_kernel_f(f, kscale, meshes[m], bundles[b], mem1, mem2, 42.);
          AURA_BENCHMARK(run_kernel_f(f, kscale, meshes[m], bundles[b], 
            mem1, mem2, 42.), runtime, min, max, mean, stdev, runs);
          std::cout << ops_tbl[3] << " (" << vsize << ") mesh (" << 
            meshes[m] << ") bundle (" << bundles[b] << ") min " << min << 
            " max " << max << " mean " << mean << " stdev " << stdev << 
            " runs " << runs << " runtime " << runtime << std::endl;
        }
        if(ops[4]) { // add 
          run_kernel(f, kadd, meshes[m], bundles[b], mem1, mem2, mem3);
          AURA_BENCHMARK(run_kernel(f, kadd, meshes[m], bundles[b], 
            mem1, mem2, mem3), runtime, min, max, mean, stdev, runs);
          std::cout << ops_tbl[4] << " (" << vsize << ") mesh (" << 
            meshes[m] << ") bundle (" << bundles[b] << ") min " << min << 
            " max " << max << " mean " << mean << " stdev " << stdev << 
            " runs " << runs << " runtime " << runtime << std::endl;
        }
        if(ops[5]) { // triad 
          run_kernel_f(f, ktriad, meshes[m], bundles[b], mem1, mem2, mem3, 42.);
          AURA_BENCHMARK(run_kernel_f(f, ktriad, meshes[m], bundles[b], mem1, 
            mem2, mem3, 42.), runtime, min, max, mean, stdev, runs);
          std::cout << ops_tbl[5] << " (" << vsize << ") mesh (" << 
            meshes[m] << ") bundle (" << bundles[b] << ") min " << min << 
            " max " << max << " mean " << mean << " stdev " << stdev << 
            " runs " << runs << " runtime " << runtime << std::endl;
        }
      }
      device_free(mem1, d);
      device_free(mem2, d);
      device_free(mem3, d);
    }
  }


  if(ops[6] || ops[7]) {
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
      device_free(m, d);
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
