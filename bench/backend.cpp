// run backend benchmarks:
//
// * create context
// * create context and feed
// * synchronization only
// * synchronization with kernel launch
// * kernel launch without synchronization
// * empty kernel with varying number of 
//   parameters, mesh and bundle size

#include <vector>
#include <tuple>
#include <bitset>
#include <boost/aura/detail/svec.hpp>
#include <boost/aura/misc/sequence.hpp>
#include <boost/aura/backend.hpp>
#include <boost/aura/misc/benchmark.hpp>


const char * ops_tbl[] = { "sync", "synck", "kern", 
  "params", "ctx", "ctxfeed" };

using namespace boost::aura;
using namespace boost::aura::backend;

#if AURA_BACKEND_OPENCL
const char * kernel_file = "bench/backend.cl"; 
#elif AURA_BACKEND_CUDA
const char * kernel_file = "bench/backend.ptx"; 
#endif

inline void run_ctx(int dev_ordinal) {
  device * d = new device(dev_ordinal);
  delete d;
}

inline void run_ctxfeed(int dev_ordinal) {
  device * d = new device(dev_ordinal);
  feed f(*d);
  delete d;
}

inline void run_sync(feed & f) {
  wait_for(f);
}

inline void run_synck(feed & f, kernel & k) {
  invoke(k, mesh(1), bundle(1), f);
  wait_for(f);
}

inline void run_kern(feed & f, kernel & k) {
  invoke(k, mesh(1), bundle(1), f);
}

inline void run_params(feed & f, kernel & k, 
  svec<std::size_t, AURA_MAX_MESH_DIMS> & mesh,
  svec<std::size_t, AURA_MAX_BUNDLE_DIMS> & bundle, 
  std::size_t params) {
  float * f1=nullptr, * f2=nullptr, * f3=nullptr, * f4=nullptr, 
    * f5=nullptr, * f6=nullptr, * f7=nullptr, * f8=nullptr, 
    * f9=nullptr, * f10=nullptr;
  switch(params) {
    case 0:
      invoke(k, mesh, bundle, f);
      break;
    case 1:
      invoke(k, mesh, bundle, args(f1), f);
      break;
    case 2:
      invoke(k, mesh, bundle, args(f1, f2), f);
      break;
    case 3:
      invoke(k, mesh, bundle, args(f1, f2, f3), f);
      break;
    case 4:
      invoke(k, mesh, bundle, args(f1, f2, f3, f4), f);
      break;
    case 5:
      invoke(k, mesh, bundle, args(f1, f2, f3, f4, f5), f);
      break;
    case 6:
      invoke(k, mesh, bundle, args(f1, f2, f3, f4, f5, f6), f);
      break;
    case 7:
      invoke(k, mesh, bundle, args(f1, f2, f3, f4, f5, f6, f7), f);
      break;
    case 8:
      invoke(k, mesh, bundle, args(f1, f2, f3, f4, f5, f6, f7, f8), f);
      break;
    case 9:
      invoke(k, mesh, bundle, args(f1, f2, f3, f4, f5, f6, f7, f8, f9), f);
      break;
    case 10:
      invoke(k, mesh, bundle, args(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10), f);
      break;
  }
  wait_for(f);
}

inline void run_tests(
  std::vector<svec<std::size_t, AURA_MAX_MESH_DIMS> > & meshes, 
  std::vector<svec<std::size_t, AURA_MAX_BUNDLE_DIMS> > & bundles,
  std::vector<svec<std::size_t, 1> > & params,
  int dev_ordinal, std::size_t runtime,
  std::bitset< sizeof(ops_tbl)/sizeof(ops_tbl[0]) > & ops) {

  // benchmark result variables
  double min, max, mean, stdev;
  std::size_t runs;
 
  // benchmarks working with device and feed 
  if(ops[4]) { // ctx
    run_ctx(dev_ordinal); // dry run
    AURA_BENCHMARK(run_ctx(dev_ordinal), runtime, min, max, mean, stdev, runs);
    print_benchmark_results(ops_tbl[4], min, max, mean, stdev, runs, runtime);
  }
  
  if(ops[5]) { // ctxfeed
    run_ctxfeed(dev_ordinal); // dry run
    AURA_BENCHMARK(run_ctxfeed(dev_ordinal), runtime, min, max, mean, stdev, runs);
    print_benchmark_results(ops_tbl[5], min, max, mean, stdev, runs, runtime);
  }

  // benchmarks requiring device and feed 
  {
    device d(dev_ordinal);
    feed f(d);

    module m = create_module_from_file(kernel_file, d, 
      AURA_BACKEND_COMPILE_FLAGS);
    
    
    if(ops[0]) { // sync
      run_sync(f); // dry run
      AURA_BENCHMARK(run_sync(f), runtime, min, max, mean, stdev, runs);
      print_benchmark_results(ops_tbl[0], min, max, mean, stdev, runs, runtime);
    }
    if(ops[1]) { // synck 
      kernel nak = create_kernel(m, "kernel_0arg");
      run_synck(f, nak); // dry run
      AURA_BENCHMARK(run_synck(f, nak), runtime, min, max, mean, stdev, runs);
      print_benchmark_results(ops_tbl[1], min, max, mean, stdev, runs, runtime);
    }
    if(ops[2]) { // kern 
      kernel nak = create_kernel(m, "kernel_0arg");
      run_kern(f, nak); // dry run
      AURA_BENCHMARK(run_kern(f, nak), runtime, min, max, mean, stdev, runs);
      print_benchmark_results(ops_tbl[2], min, max, mean, stdev, runs, runtime);
      wait_for(f); // we did not synchronize in this benchmark
    }
    if(ops[3]) { // params
      for(std::size_t p=0; p<params.size(); p++) {
        
        char kernel_name[] = "kernel_XXXarg";
        assert(0 <= params[p][0] && 11 > params[p][0]);
        snprintf(kernel_name, sizeof(kernel_name)-1, 
          "kernel_%luarg", params[p][0]);
        kernel k = create_kernel(m, kernel_name);
        
        for(std::size_t m=0; m<meshes.size(); m++) {
          for(std::size_t b=0; b<bundles.size(); b++) {
            run_params(f, k, meshes[m], bundles[b], params[p][0]);      
            AURA_BENCHMARK(run_params(f, k, meshes[m], bundles[b], 
              params[p][0]), runtime, min, max, mean, stdev, runs);
            std::cout << ops_tbl[6] << "m (" << meshes[m] << ") b (" << 
              bundles[m] << ") p " << params[p][0] << " min " << min << 
              " max " << max << " mean " << mean << " stdev " << stdev << 
              " runs " << runs << " runtime " << runtime << std::endl;
          }
        }
      }
    }
  }
  
}


int main(int argc, char *argv[]) {

  initialize();
  
  // parse command line arguments:
  // -m mesh sizes (sequence, max rank 3)
  // -b bundle sizes (sequence, max rank 3)
  // -p number of params (sequence, range 0-10)
  // -d device (single value)
  // -t time (time per benchmark in ms)

  // config params
  std::bitset< sizeof(ops_tbl)/sizeof(ops_tbl[0]) > ops;
  std::vector<svec<std::size_t, AURA_MAX_MESH_DIMS> > meshes;
  std::vector<svec<std::size_t, AURA_MAX_BUNDLE_DIMS> > bundles;
  std::vector<svec<std::size_t, 1> > params;
  int dev_ordinal = 0;
  std::size_t runtime = 0;
 
  // parse config
  int opt;
  while ((opt = getopt(argc, argv, "m:b:p:d:t:")) != -1) {
    switch (opt) {
      case 'm': {
        printf("mesh: %s ", optarg);
        meshes = boost::aura::generate_sequence<std::size_t, 
               AURA_MAX_MESH_DIMS>(optarg);
        break;
      }
      case 'b': {
        printf("bundle: %s ", optarg);
        bundles = boost::aura::generate_sequence<std::size_t, 
                AURA_MAX_BUNDLE_DIMS>(optarg);
        break;
      }
      case 'p': {
        printf("param: %s ", optarg);
        params = boost::aura::generate_sequence<std::size_t, 1>(optarg);
        break;
      }
      case 'd': {
        printf("device %s ", optarg);
        dev_ordinal = atoi(optarg);
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
        fprintf(stderr, "Usage: %s -m <meshsize> -b <bundlesize> -p <params> "
          "-d <device ordinal> -t <runtime (ms)> <operations>\n", argv[0]);
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
    device d(dev_ordinal);
    device_info di = device_get_info(d);
    printf("selected device: ");
    print_device_info(di); 
  }
  run_tests(meshes, bundles, params, dev_ordinal, runtime, ops);

}


