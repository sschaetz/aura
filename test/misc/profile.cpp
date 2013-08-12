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
  for(int i=0; i<10; i++) {
    profile::start(s, "foo");
    usleep(10000);
    profile::stop(s, "foo");
    usleep(1000);
  }
}

void runner1(profile::memory_sink & s) {
  AURA_PROFILE_FUNCTION(profile::memory_sink, s);
  profile::start(s, "runner1 subtask");
  usleep(20000);
  profile::start(s, "runner1 subsubtask");
  usleep(20000);
  profile::stop(s, "runner1 subsubtask");
  profile::stop(s, "runner1 subtask");
  usleep(20000);
}

BOOST_AUTO_TEST_CASE(basic) {
  profile::memory_sink s;
  {
    AURA_PROFILE_FUNCTION(profile::memory_sink, s);
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
  }
  s.dump("/tmp/profile.log");
  profile::dump_svg(s, "/home/sschaet/profile.svg");
}
