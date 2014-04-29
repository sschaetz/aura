
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <bitset>
#include <tuple>
#include <complex>

#include <aura/misc/sequence.hpp>
#include <aura/misc/benchmark.hpp>
#include <aura/backend.hpp>

using namespace aura;
using namespace aura::backend;

typedef std::complex<float> cfloat;

// FIXME missing type (double, float) and r2c c2r

// configuration
std::vector<bounds> size;
std::vector<aura::svec<std::size_t, 1> > batch;
std::size_t runtime;

aura::svec<std::size_t> devordinals;

const char * ops_tbl[] = { "fwdip", "invip", "fwdop", "invop" };
std::bitset< sizeof(ops_tbl)/sizeof(ops_tbl[0]) > ops;

// we have multiple wait_for free functions, std::for_each
// can not decide which one should be used
void wait_for_feed(feed& f) {
	wait_for(f);
}

// benchmark functions -----

void run_fwdip(std::vector<device_ptr<cfloat> > & mem1, 
    std::vector<fft> & ffth, std::vector<feed> & feeds) {
  for(std::size_t n = 0; n<feeds.size(); n++) {
    fft_forward(mem1[n], mem1[n], ffth[n], feeds[n]);
  }
  std::for_each(feeds.begin(), feeds.end(), &wait_for_feed);
}

void run_invip(std::vector<device_ptr<cfloat> > & mem1, 
    std::vector<fft> & ffth, std::vector<feed> & feeds) {
  for(std::size_t n = 0; n<feeds.size(); n++) {
    fft_inverse(mem1[n], mem1[n], ffth[n], feeds[n]);
  }
  std::for_each(feeds.begin(), feeds.end(), &wait_for_feed);
}

void run_fwdop(std::vector<device_ptr<cfloat> > & mem1, 
    std::vector<device_ptr<cfloat> > & mem2, 
    std::vector<fft> & ffth, 
    std::vector<feed> & feeds) {
  for(std::size_t n = 0; n<feeds.size(); n++) {
    fft_forward(mem1[n], mem2[n], ffth[n], feeds[n]);
  }
  std::for_each(feeds.begin(), feeds.end(), &wait_for_feed);
}

void run_invop(std::vector<device_ptr<cfloat> > & mem1, 
    std::vector<device_ptr<cfloat> > & mem2, 
    std::vector<fft> & ffth, 
    std::vector<feed> & feeds) {
  for(std::size_t n = 0; n<feeds.size(); n++) {
    fft_inverse(mem1[n], mem2[n], ffth[n], feeds[n]);
  }
  std::for_each(feeds.begin(), feeds.end(), &wait_for_feed);
}

// -----

void print_results(const char * name, double min, double max, 
    double mean, double stdev, std::size_t runs,
    const bounds & s,
    const aura::svec<std::size_t, 1> & batch) {
  printf("%s %lux ", name, batch[0]);
  for(std::size_t i=0; i<s.size(); i++) {
    printf("%lu ", s[i]);
  }
  printf("min %f max %f mean %f stdev %f runs %lu\n", 
    min, max, mean, stdev, runs);
}

void run_tests() {
  aura::backend::initialize();
  aura::backend::fft_initialize(); 
  
  // create devices, feeds
  std::vector<aura::backend::device> devices;
  std::vector<aura::backend::feed> feeds;
  // reserve to make sure the device objects are not moved
  devices.reserve(devordinals.size());
  feeds.reserve(devordinals.size());
  for(std::size_t i=0; i<devordinals.size(); i++) {
    devices.push_back(aura::backend::device(devordinals[i]));
    feeds.push_back(aura::backend::feed(devices[i]));
  }
  
  for(std::size_t b=0; b<batch.size(); b++) {
    for(std::size_t s=0; s<size.size(); s++) {
      // allocate device_ptr<cfloat> , make fft plan
      std::vector<aura::backend::device_ptr<cfloat> > mem1;
      std::vector<aura::backend::device_ptr<cfloat> > mem2;
      std::vector<aura::backend::fft> ffth;
      for(std::size_t i=0; i<devices.size(); i++) {
        std::size_t msize = aura::product(size[s]) * batch[b][0];
        mem1.push_back(aura::backend::device_malloc<cfloat>(msize, devices[i]));
        mem2.push_back(aura::backend::device_malloc<cfloat>(msize, devices[i]));
        ffth.push_back(aura::backend::fft(devices[i], feeds[i], size[s], 
          aura::backend::fft::type::c2c, batch[b][0]));
      }
      
      // benchmark result variables
      double min, max, mean, stdev;
      std::size_t runs;
      
      if(ops[0]) {
        run_fwdip(mem1, ffth, feeds);        
        AURA_BENCHMARK(run_fwdip(mem1, ffth, feeds), 
          runtime, min, max, mean, stdev, runs);
        print_results(ops_tbl[0], min, max, mean, stdev, runs, 
          size[s], batch[b]);
      }
      if(ops[1]) {
        run_invip(mem1, ffth, feeds);        
        AURA_BENCHMARK(run_invip(mem1, ffth, feeds), 
          runtime, min, max, mean, stdev, runs);
        print_results(ops_tbl[1], min, max, mean, stdev, runs,
          size[s], batch[b]);
      }
      if(ops[2]) {
        run_fwdop(mem1, mem2, ffth, feeds);        
        AURA_BENCHMARK(run_fwdop(mem1, mem2, ffth, feeds), 
          runtime, min, max, mean, stdev, runs);
        print_results(ops_tbl[2], min, max, mean, stdev, runs,
          size[s], batch[b]);
      }
      if(ops[3]) {
        run_invop(mem1, mem2, ffth, feeds);        
        AURA_BENCHMARK(run_invop(mem1, mem2, ffth, feeds), 
          runtime, min, max, mean, stdev, runs);
        print_results(ops_tbl[3], min, max, mean, stdev, runs,
          size[s], batch[b]);
      }

      // free device_ptr<cfloat>  
      for(std::size_t i=0; i<devices.size(); i++) {
        aura::backend::device_free(mem1[i]);
        aura::backend::device_free(mem2[i]);
      }
    }
  }
  aura::backend::fft_terminate();
}

int main(int argc, char *argv[]) {

  // parse command line arguments:
  // the vector size -s, the batch size -b (both sequences)
  // the runtime per test -t in ms
  // and a list of device ordinals
  // and the options: fwdip, invip, fwdop, invop 
  
  int opt;
  while ((opt = getopt(argc, argv, "s:b:t:d:")) != -1) {
    switch (opt) {
      case 's': {
        printf("size: %s ", optarg);
	size = aura::generate_sequence<int, AURA_SVEC_MAX_SIZE>(optarg);
        printf("(%lu) ", size.size());
        break;
      }
      case 't': {
        runtime = atoi(optarg);
        printf("time: %lu ms ", runtime);
        // benchmark script expects us
        runtime *= 1000; 
        break;
      }
      case 'b': {
        printf("batch: %s ", optarg);
        batch = aura::generate_sequence<std::size_t, 1>(optarg);
        printf("(%lu) ", batch.size());
        break;
      }
      case 'd': {
        char * optarg_copy = optarg;
        while(true)
        {
          char * cur = strsep(&optarg_copy, ",");
          if(cur == NULL) {
            break;
          }
          devordinals.push_back(atoi(cur));
        }
        break;
      }
      default: {
        fprintf(stderr, "Usage: %s -s <vectorsize> -b <batchsize> -t <runtime> "
          "<operations>\n", argv[0]);
        exit(-1);
      }
    }
  }
  printf("options: ");
  for(unsigned int i=0; i<sizeof(ops_tbl)/sizeof(ops_tbl[0]); i++) {
    ops[i] = false;
    for(int j=optind; j<argc; j++) {
      if(NULL != strstr(argv[j], ops_tbl[i])) {
        printf("%s ", ops_tbl[i]);
        ops[i] = true;
      }
    }
  }
  printf("\ndevices: ");
  for(unsigned int i=0; i<devordinals.size(); i++) {
    printf("%lu ", devordinals[i]);
  }
  printf("\nepxected runtime %1.2fs\n", 
    batch.size()*size.size()*runtime*ops.count()/1000./1000.);

  
  run_tests();

}


