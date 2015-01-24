#ifndef AURA_MISC_DEPRECATE_HPP
#define AURA_MISC_DEPRECATE_HPP

// code taken from http://stackoverflow.com/a/295229
// to deprecate a function
//
// since the API changes, this can help to dected use of
// old interfaces at compile time

#ifdef __APPLE__
// TODO deprecation for Mac OS
#define DEPRECATED(func) func 
#else
#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED(func) func
#endif
#endif

#endif
