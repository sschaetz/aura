
IF(${AURA_BACKEND} STREQUAL OPENCL)
  AURA_ADD_TARGET(exp.context_opencl context_opencl.cpp 
    ${AURA_BACKEND_LIBRARIES} ${Boost_SYSTEM_LIBRARY} 
    ${Boost_THREAD_LIBRARY} pthread rt)
  AURA_ADD_TARGET(exp.oclpinning oclpinning.cpp 
    ${AURA_BACKEND_LIBRARIES})
ENDIF()

IF(NOT ${AURA_FFT_LIBRARIES} STREQUAL "")
  AURA_ADD_TARGET(exp.fft_2d_1d fft_2d_1d.cpp 
    ${AURA_BACKEND_LIBRARIES} ${AURA_FFT_LIBRARIES})
ENDIF()

IF(${AURA_BACKEND} STREQUAL CUDA)
  AURA_ADD_TARGET(exp.fftsweep_cuda fftsweep_cuda.cpp 
    ${AURA_BACKEND_LIBRARIES} ${AURA_FFT_LIBRARIES})
  AURA_ADD_TARGET(exp.bughunt bughunt.cpp 
    ${AURA_BACKEND_LIBRARIES} ${AURA_FFT_LIBRARIES})
  AURA_ADD_TARGET(exp.overlap overlap.cpp overlap.cu 
    ${AURA_BACKEND_LIBRARIES} ${AURA_FFT_LIBRARIES})
ENDIF()

AURA_ADD_TARGET(exp.namespace_alias namespace_alias.cpp)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")   
AURA_ADD_TARGET(exp.vec_vs_vla vec_vs_vla.cpp)
AURA_ADD_TARGET(exp.view view.cpp)
AURA_ADD_TARGET(exp.viewv2 viewv2.cpp)

IF(NOT ${AURA_FFTW_LIBRARY} STREQUAL "")
	AURA_ADD_TARGET(exp.future future.cpp pthread fftw3f ${AURA_FFTW_LIBRARY}
		${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY})
ENDIF()
AURA_ADD_TARGET(exp.ftrftr ftrftr.cpp pthread fftw3f ${AURA_FFTW_LIBRARY}
	${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY})

