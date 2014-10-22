CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# configuration #####
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda/" CACHE STRING "CUDA path")

# choose CUDA, OPENCL
IF(NOT AURA_BACKEND)
  SET(AURA_BACKEND OPENCL)
ENDIF()

# make additional modules available and source custom functions 
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
INCLUDE(cmake/functions.cmake)


# find Boost libraries #####
OPTION(BOOST_TEST_DYN_LINK "Link tests against dynamic version of boost unittest library" ON)
IF (WIN32)
    SET(Boost_USE_STATIC_LIBS ON)
ELSE ()
  IF (BOOST_TEST_DYN_LINK)
    ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
  ENDIF ()
ENDIF()
SET(BOOST_ROOT ${BOOST_ROOT} "/usr/local/boost")
FIND_PACKAGE(Boost COMPONENTS system thread unit_test_framework)

# find CUDA libraries #####
FIND_PACKAGE(CUDA)

# find OpenCL libraries #####
FIND_PACKAGE(OpenCL)


# sanity check found libraries and selected backend
IF(${AURA_BACKEND} STREQUAL CUDA)
  IF(NOT ${CUDA_FOUND})
    MESSAGE(FATAL_ERROR "CUDA backend selected but CUDA could not be found")
  ELSE()
    SET(AURA_BACKEND_LIBRARIES ${CUDA_LIBRARIES} libcuda.so)
    SET(AURA_BACKEND_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    SET(AURA_FFT_LIBRARIES ${CUDA_CUFFT_LIBRARIES})
    SET(AURA_FFT_CUFFT 1)
    SET(AURA_BACKEND_CUDA 1)
    SET(AURA_BACKEND_OPENCL 0)
    LINK_DIRECTORIES(${LINK_DIRECTORIES} /usr/lib/arm-linux-gnueabihf/tegra)
  ENDIF()
ENDIF()
IF(${AURA_BACKEND} STREQUAL OPENCL)
  IF(NOT ${OPENCL_FOUND})
    MESSAGE(FATAL_ERROR "OpenCL backend selected but OpenCL could not be found")
  ELSE()
    SET(AURA_BACKEND_LIBRARIES ${OPENCL_LIBRARIES})
    SET(AURA_BACKEND_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
 
    FIND_PACKAGE(clFFT)
    IF(${CLFFT_FOUND})
      SET(AURA_FFT_LIBRARIES ${CLFFT_LIBRARIES})
      INCLUDE_DIRECTORIES(${CLFFT_INCLUDE_DIRS})
      SET(AURA_FFT_CLFFT 1)
    ELSE()
      SET(AURA_FFT_LIBRARIES "") 
      SET(AURA_FFT_CLFFT 0)
    ENDIF()
    
    SET(AURA_BACKEND_CUDA 0)
    SET(AURA_BACKEND_OPENCL 1)
  ENDIF()
ENDIF()
STRING(TOLOWER "${AURA_BACKEND}" AURA_BACKEND_LC)
MESSAGE(STATUS "Using ${AURA_BACKEND} backend")

# compiler settings
SET(CMAKE_CXX_FLAGS "-Wall -std=c++0x -g")

# used for OpenCL kernels that need to have the include dirs at runtime
SET(AURA_BACKEND_COMPILE_FLAGS 
  "-I${PROJECT_SOURCE_DIR}/include/ -I${CMAKE_BINARY_DIR}")

# used for unittests to find the kernels
SET(AURA_UNIT_TEST_LOCATION "${CMAKE_BINARY_DIR}/test/")

# configuration file
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/include/aura/config.hpp.template"
  "${CMAKE_BINARY_DIR}/aura/config.hpp")

# set global include directories for project 
INCLUDE_DIRECTORIES("${AURA_BACKEND_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include/")
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}/")
INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}/")

# enable testing
ENABLE_TESTING()

# add subdirectories
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/experiment/)
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/test/)
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/bench/)

