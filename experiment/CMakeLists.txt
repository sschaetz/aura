
IF(${AURA_BACKEND} STREQUAL OPENCL)
  AURA_ADD_TARGET(exp.context_opencl context_opencl.cpp 
    ${AURA_BACKEND_LIBRARIES} ${Boost_THREAD_LIBRARY} 
    ${Boost_SYSTEM_LIBRARY} pthread)
ENDIF()

