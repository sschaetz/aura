#define BOOST_TEST_MODULE backend.fft

#include <complex>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>
#include <aura/config.hpp>

typedef std::complex<float> cfloat;

using namespace aura::backend;

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
  init();
  fft_init(); 
  int num = device_get_count();
  if(0 < num) {
    int samples = 4;
    const cfloat signal[] = 
      {cfloat(1, 1), cfloat(2, 2), cfloat(3, 3), cfloat(4, 4)};
    const cfloat spectrum[] = 
      {cfloat(10, 10), cfloat(-4, 0), cfloat(-2, -2), cfloat(0, -4)};
    assert(samples == sizeof(signal) / sizeof(signal[0]));
    
    std::vector<cfloat> input(signal, signal+samples);
    std::vector<cfloat> output(samples, cfloat(555., 666.));
    
    device d(0);
    feed f(d); 
    
    memory m1 = device_malloc(samples*sizeof(cfloat), d);
    memory m2 = device_malloc(samples*sizeof(cfloat), d);
    copy(m1, &input[0], samples*sizeof(cfloat), f);
    copy(m2, &output[0], samples*sizeof(cfloat), f);
    
    fft fh(d, fft_dim(samples), fft::type::c2c);
    fft_forward(m2, m1, fh, f);
    
    copy(&output[0], m2, samples*sizeof(cfloat), f);
    wait_for(f);
    BOOST_CHECK(std::equal(output.begin(), output.end(), spectrum));
    device_free(m1, d);
    device_free(m2, d);
  }
  fft_finish();
}

