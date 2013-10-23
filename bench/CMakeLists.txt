AURA_ADD_TARGET(bench.micro micro.cpp micro.cl micro.cu 
  ${AURA_BACKEND_LIBRARIES})

IF(${AURA_BACKEND} STREQUAL "CUDA")
AURA_ADD_TARGET(bench.overlap overlap.cpp overlap.cu 
  ${AURA_BACKEND_LIBRARIES} ${AURA_FFT_LIBRARIES})
ENDIF()

