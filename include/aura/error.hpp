#ifndef AURA_ERROR_HPP
#define AURA_ERROR_HPP

namespace aura {

#define AURA_ERROR(error) { \
  fprintf(stderr, "ERROR: %s\n in %s:%d", \
    error, __FILE__, __LINE__); \
	abort(); \
} \
/**/

} // arua

#endif // AURA_STREAM_HPP

