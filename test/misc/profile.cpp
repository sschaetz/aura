#define BOOST_TEST_MODULE misc.profile

#include <unistd.h>
#include <stdio.h>
#include <boost/test/unit_test.hpp>
#include <aura/misc/profile.hpp>
#include <aura/misc/profile_svg.hpp>

using namespace aura;

// basic
// _____________________________________________________________________________

void runner0(profile::memory_sink & s) {
  profile::start(s, "foo");
  usleep(100000);
  profile::stop(s, "foo");
}

void runner1(profile::memory_sink & s) {
  AURA_PROFILE_FUNCTION(profile::memory_sink, s);
  usleep(200000);
}

BOOST_AUTO_TEST_CASE(basic) {
  profile::memory_sink s;
  profile::start(s, "basic");
  profile::stop(s, "basic");
  std::thread t0(runner0, std::ref(s));
  profile::start(s, "basic");
  profile::stop(s, "basic");
  std::thread t1(runner1, std::ref(s));
  profile::start(s, "basic");
  profile::stop(s, "basic");
  t0.join();
  t1.join();
  s.dump("/tmp/profile.log");
  profile::dump_svg(s, "/home/sschaet/profile.svg");
}
