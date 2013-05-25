CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# configuration #####

# choose CUDA, OPENCL
IF(NOT AURA_BACKEND)
  SET(AURA_BACKEND OPENCL)
ENDIF()

# make additional modules available and source custom functions 
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
INCLUDE(cmake/functions.cmake)


# find Boost libraries #####
SET(Boost_USE_STATIC_LIBS ON)
SET(Boost_ADDITIONAL_VERSIONS "1.47" "1.47.0" "1.48" "1.48.0")
FIND_PACKAGE(Boost 1.45.0 COMPONENTS thread unit_test_framework)


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
    SET(AURA_BACKEND_CUDA 1)
    SET(AURA_BACKEND_OPENCL 0)
  ENDIF()
ENDIF()
IF(${AURA_BACKEND} STREQUAL OPENCL)
  IF(NOT ${OPENCL_FOUND})
    MESSAGE(FATAL_ERROR "OpenCL backend selected but OpenCL could not be found")
  ELSE()
    SET(AURA_BACKEND_LIBRARIES ${OPENCL_LIBRARIES})
    SET(AURA_BACKEND_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
    SET(AURA_BACKEND_CUDA 0)
    SET(AURA_BACKEND_OPENCL 1)
  ENDIF()
ENDIF()
STRING(TOLOWER "${AURA_BACKEND}" AURA_BACKEND_LC)
MESSAGE(STATUS "Using ${AURA_BACKEND} backend")

# configuration file
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/include/aura/config.hpp.template"
  "${PROJECT_SOURCE_DIR}/include/aura/config.hpp")

# set global include directories for project 
INCLUDE_DIRECTORIES("${AURA_BACKEND_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include/")

# enable testing
ENABLE_TESTING()

# add subdirectories
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/experiment/)
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/test/)
