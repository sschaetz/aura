#ifndef AURA_MISC_NOW_HPP
#define AURA_MISC_NOW_HPP

#include <boost/config.hpp>

#ifdef BOOST_HAS_GETTIMEOFDAY

#include <sys/time.h>

namespace aura {

inline double now() {
  struct timeval tp;
  gettimeofday(&tp ,NULL);
  return double(tp.tv_sec)*1e6 + double(tp.tv_usec);
}

} // namespace aura

#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

#include <windows.h>

namespace aura {

inline double now() {
  union pli { LARGE_INTEGER li; double d; };
  pli freq,t;
  QueryPerformanceFrequency( &freq.li );
  QueryPerformanceCounter( &t.li );
  return (((t.d*1000000000)/freq.d)/1000000000);
}

} // namespace aura


#else

#include <ctime>

namespace aura {
  inline double now() { return std::clock()/double(CLOCKS_PER_SEC); }
} // namespace aura

#endif // defined(BOOST_HAS_GETTIMEOFDAY)

#endif // AURA_MISC_NOW_HPP

