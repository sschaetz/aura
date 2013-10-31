
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <bitset>
#include <tuple>
#include <complex>

#include <aura/misc/sequence.hpp>
#include <aura/backend.hpp>

typedef std::complex<float> cfloat;

// configuration
aura::sequence<3> size;
aura::sequence<1> batch;
std::size_t runtime;

aura::svec<std::size_t> devordinals;

const char * ops_tbl[] = { "fwdip", "bwdip", "fwdop", "bwdop" };
std::bitset< sizeof(ops_tbl)/sizeof(ops_tbl[0]) > ops;


void run_tests() {
  aura::backend::initialize();
  aura::backend::fft_initialize(); 
  
  // create devices, feeds
  std::vector<aura::backend::device> devices;
  std::vector<aura::backend::feed> feeds;
  devices.reserve(devordinals.size());
  feeds.reserve(devordinals.size());
  for(std::size_t i=0; i<devordinals.size(); i++) {
    devices.push_back(aura::backend::device(devordinals[i]));
    feeds.push_back(aura::backend::feed(devices[i]));
  }
  
  aura::svec<std::size_t, 1> b;
  bool bgood;
  std::tie(b, bgood) = batch.next();
 
  while(bgood) {
    aura::svec<std::size_t, 3> s;
    bool sgood;
    std::tie(s, sgood) = size.next();
    while(sgood) {
      // allocate memory, make fft plan
      std::vector<aura::backend::memory> mem1;
      std::vector<aura::backend::memory> mem2;
      std::vector<aura::backend::fft> ffth;
      for(std::size_t i=0; i<devices.size(); i++) {
        std::size_t msize = aura::product(s) * b[0] * sizeof(cfloat);
        mem1.push_back(aura::backend::device_malloc(msize, devices[i])); 
        mem2.push_back(aura::backend::device_malloc(msize, devices[i]));
        printf("memsize %lu\n", msize);
        // handle type mismatch int <-> std::size_t
        aura::backend::fft_dim d;
        for(unsigned long int i=0; i<s.size(); i++) {
          d.push_back((int)s[i]);
        }
        ffth.push_back(aura::backend::fft(devices[i], d, 
          aura::backend::fft::type::c2c, b[0]));
      }
      // dry run fft
      // run benchmark

      
      for(std::size_t i=0; i<devices.size(); i++) {
        aura::backend::device_free(mem1[i], devices[i]);
        aura::backend::device_free(mem2[i], devices[i]);
      }
      std::tie(s, sgood) = size.next();
    }
    size.rewind();
    std::tie(b, bgood) = batch.next();
  }
  
}

int main(int argc, char *argv[]) {

  // parse command line arguments:
  // the vector size -s, the batch size -b (both sequences)
  // the runtime per test -t in ms
  // and a list of device ordinals
  // and the options: fwd, bwd, ip, op (in-place, out-of-place)
  
  int opt;
  while ((opt = getopt(argc, argv, "s:b:t:d:")) != -1) {
    switch (opt) {
      case 's': {
        printf("size: %s ", optarg);
        size = aura::sequence<3>(optarg);
        printf("(%lu) ", size.size());
        break;
      }
      case 't': {
        runtime = atoi(optarg);
        printf("time: %lu ms ", runtime);
        break;
      }
      case 'b': {
        printf("batch: %s ", optarg);
        batch = aura::sequence<1>(optarg);
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
    batch.size()*size.size()*runtime*ops.count()/1000.);

  
  run_tests();

}


