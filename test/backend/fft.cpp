#define BOOST_TEST_MODULE backend.fft

#include <complex>
#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>
#include <aura/config.hpp>

typedef std::complex<float> cfloat;

using namespace aura::backend;

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
  initialize();
  fft_initialize(); 
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
    
    device_ptr<cfloat> m1 = device_malloc<cfloat>(samples, d);
    device_ptr<cfloat> m2 = device_malloc<cfloat>(samples, d);
    copy(m1, &input[0], samples, f);
    copy(m2, &output[0], samples, f);
    
    fft fh(d, f, fft_dim(samples), fft::type::c2c);
    fft_forward(m2, m1, fh, f);
    
    copy(&output[0], m2, samples, f);
    wait_for(f);
    BOOST_CHECK(std::equal(output.begin(), output.end(), spectrum));
    device_free(m1);
    device_free(m2);
  }
  fft_terminate();
}

