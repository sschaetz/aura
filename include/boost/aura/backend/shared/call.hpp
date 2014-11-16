#ifndef AURA_BACKEND_SHARED_CALL_HPP
#define AURA_BACKEND_SHARED_CALL_HPP

#define AURA_CHECK_ERROR(expr) { \
  if (!expr) { \
    printf("AURA error at %s:%d\n", __FILE__, __LINE__ ); \
  } \
} \
/**/

#endif // AURA_BACKEND_SHARED_CALL_HPP

