AURA_ADD_TEST(backend/basic.cpp ${AURA_BACKEND_LIBRARIES})
AURA_ADD_TEST(backend/device_info.cpp ${AURA_BACKEND_LIBRARIES})
AURA_ADD_TEST(backend/feed.cpp ${AURA_BACKEND_LIBRARIES})

IF(NOT ${AURA_FFT_LIBRARIES} STREQUAL "")
  AURA_ADD_TEST(backend/fft.cpp pthread
    ${AURA_BACKEND_LIBRARIES} ${AURA_FFT_LIBRARIES})
ENDIF()

AURA_ADD_TEST(backend/kernel.cpp backend/kernel.cu 
  backend/kernel.cl ${AURA_BACKEND_LIBRARIES})

AURA_ADD_TEST(detail/svec.cpp)
AURA_ADD_TEST(misc/profile.cpp pthread)

# cprofile is a C interface for the profiler, it requires
# a C++ library that exports C functions
ADD_LIBRARY(cprofile ${PROJECT_SOURCE_DIR}/source/misc/cprofile.cpp)
AURA_ADD_TEST(misc/cprofile.c cprofile pthread)
AURA_ADD_TEST(misc/sequence.cpp)

