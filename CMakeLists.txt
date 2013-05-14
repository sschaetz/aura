CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# configuration #####

# choose CUDA, OPENCL
SET(AURA_BACKEND CUDA)


# make additional modules available
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")


# find Boost libraries #####
#SET(Boost_USE_STATIC_LIBS ON)
#SET(Boost_ADDITIONAL_VERSIONS "1.47" "1.47.0")
#FIND_PACKAGE(Boost 1.45.0 COMPONENTS thread unit_test_framework)


# find CUDA libraries #####
FIND_PACKAGE(CUDA)


# find OpenCL libraries #####
FIND_PACKAGE(OpenCL)


# sanity check found libraries and selected backend

IF(${AURA_BACKEND} STREQUAL CUDA)
  IF(NOT ${CUDA_FOUND})
    MESSAGE(FATAL_ERROR "CUDA backend selected but CUDA could not be found")
  ELSE()
    SET(AURA_BACKEND_LIBRARIES ${CUDA_LIBRARIES})
    SET(AURA_BACKEND_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  ENDIF()
ENDIF()
IF(${AURA_BACKEND} STREQUAL OPENCL)
  IF(NOT ${OPENCL_FOUND})
    MESSAGE(FATAL_ERROR "OpenCL backend selected but OpenCL could not be found")
  ELSE()
    SET(AURA_BACKEND_LIBRARIES ${OPENCL_LIBRARIES})
    SET(AURA_BACKEND_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
  ENDIF()
ENDIF()

MESSAGE(STATUS "Using ${AURA_BACKEND} backend")



