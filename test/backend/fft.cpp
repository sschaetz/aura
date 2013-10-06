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
  int num = device_get_count();
  if(0 < num) {
    device d(0);
    feed f(d); 
    int samples = 8;
    const float signal[] = {0, 1, 1, 0, 0, 1, 1, 0};
    assert(samples == sizeof(signal) / sizeof(signal[0]));
    std::vector<float> input(signal, signal+samples);
    std::vector<cfloat> output(samples);
    memory m1 = device_malloc(samples*sizeof(float), d);
    memory m2 = device_malloc(samples*sizeof(cfloat), d);
    fft fh(d, fft_dim(8), fft::type::r2c);
    copy(m1, &input[0], samples*sizeof(float), f);
    fft_forward(m2, m1, fh, f);
    copy(&output[0], m2, samples*sizeof(cfloat), f);
    wait_for(f);
  }
}

