
AURA_ADD_TARGET(bench.backend backend.cpp backend.cu backend.cl
	${AURA_BACKEND_LIBRARIES})

AURA_ADD_TARGET(bench.micro micro.cpp micro.cl micro.cu 
	${AURA_BACKEND_LIBRARIES})

IF(NOT ${AURA_FFT_LIBRARIES} STREQUAL "")
	AURA_ADD_TARGET(bench.fft fft.cpp ${AURA_BACKEND_LIBRARIES}
		${AURA_FFT_LIBRARIES} pthread)
ENDIF()

AURA_ADD_TARGET(bench.peak peak.cpp peak.cc 
	${AURA_BACKEND_LIBRARIES})

AURA_ADD_TARGET(bench.zerocpy zerocpy.cpp zerocpy.cc  
	${AURA_BACKEND_LIBRARIES})
