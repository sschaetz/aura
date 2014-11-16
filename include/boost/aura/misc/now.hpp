#ifndef AURA_MISC_NOW_HPP
#define AURA_MISC_NOW_HPP

// beware, 
// here are ugly tricks to compile this header with c compiler

// FIXME use boost version of this thing

#ifdef __cplusplus
#include <boost/config.hpp>
#else 
#define BOOST_HAS_GETTIMEOFDAY
#endif

#ifdef BOOST_HAS_GETTIMEOFDAY

#include <sys/time.h>

#ifdef __cplusplus
namespace boost
{
namespace aura 
{
#endif

inline double now() 
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return (double)(tp.tv_sec)*1e6 + (double)(tp.tv_usec);
}

#ifdef __cplusplus
} // namespace aura
} // boost
#endif

#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

#include <windows.h>

#ifdef __cplusplus
namespace aura {
#endif

inline double now() 
{
	union pli { LARGE_INTEGER li; double d; };
	pli freq,t;
	QueryPerformanceFrequency( &freq.li );
	QueryPerformanceCounter( &t.li );
	return (((t.d*1000000000)/freq.d)/1000000000);
}

#ifdef __cplusplus
} // namespace aura
#endif


#else


#include <ctime>

#ifdef __cplusplus
namespace boost
{
namespace aura 
{
#endif

inline double now() 
{ 
	return std::clock()/double(CLOCKS_PER_SEC);
}

#ifdef __cplusplus
} // namespace aura
} // boost
#endif

#endif // #ifdef BOOST_HAS_GETTIMEOFDAY
 
#endif // AURA_MISC_NOW_HPP

