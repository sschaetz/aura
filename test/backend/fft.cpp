#define BOOST_TEST_MODULE backend.fft

#include <boost/test/unit_test.hpp>
#include <aura/backend.hpp>
#include <aura/config.hpp>

using namespace aura::backend;

// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic) {
  init();
  int num = device_get_count();
  if(0 < num) {
    device d(0);
    fft fh(d, fft_dim(512, 512), fft::type::c2c);
  }
}

