
IF(${AURA_BACKEND} STREQUAL OPENCL)
  AURA_ADD_TARGET(exp.context_opencl context_opencl.cpp 
    ${AURA_BACKEND_LIBRARIES} ${Boost_THREAD_LIBRARY} 
    pthread)
ENDIF()

IF(${AURA_BACKEND} STREQUAL CUDA)
  AURA_ADD_TARGET(exp.fftsweep_cuda fftsweep_cuda.cpp 
    ${AURA_BACKEND_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
  AURA_ADD_TARGET(exp.bughunt bughunt.cpp 
    ${AURA_BACKEND_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
ENDIF()

