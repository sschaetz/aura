CMAKE_MINIMUM_REQUIRED(VERSION 3.1)


# If no backend is selected, choose OpenCL as default. 
IF(NOT AURA_BACKEND)
        SET(AURA_BACKEND OPENCL)
ENDIF()


# Set unit test device id.
IF(NOT AURA_UNIT_TEST_DEVICE)
        SET(AURA_UNIT_TEST_DEVICE 0)
ENDIF()
ADD_DEFINITIONS(-DAURA_UNIT_TEST_DEVICE=${AURA_UNIT_TEST_DEVICE})


# Make additional modules available and source custom functions.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")


# Find Boost libraries.
OPTION(BOOST_TEST_DYN_LINK 
       "Link tests against dynamic version of boost unittest library" ON)
IF (WIN32)
        SET(Boost_USE_STATIC_LIBS ON)
ELSE ()
        IF (BOOST_TEST_DYN_LINK)
                ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
        ENDIF ()
ENDIF()
FIND_PACKAGE(Boost COMPONENTS system thread filesystem unit_test_framework)

# Dummy include dir used until backed include libs work.
SET(AURA_BACKEND_INCLUDE_DIRS ".")

#[[
# Find CUDA libraries.
FIND_PACKAGE(CUDA)

# Find OpenCL libraries
FIND_PACKAGE(OpenCL)

# Sanity check found libraries and selected backend.
IF(${AURA_BACKEND} STREQUAL CUDA)
        IF(NOT ${CUDA_FOUND})
                MESSAGE(FATAL_ERROR 
                        "CUDA backend selected but CUDA could not be found")
        ELSE()
                SET(AURA_BACKEND_LIBRARIES ${CUDA_LIBRARIES} libcuda.so)
                SET(AURA_BACKEND_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
                SET(AURA_FFT_LIBRARIES ${CUDA_CUFFT_LIBRARIES})
                SET(AURA_BACKEND_CUDA 1)
                SET(AURA_BACKEND_OPENCL 0)
                SET(AURA_BACKEND_METAL 0)
                ADD_DEFINITIONS(-DAURA_BACKEND_CUDA)
        ENDIF()
ENDIF()

IF(${AURA_BACKEND} STREQUAL OPENCL)
        IF(NOT ${OPENCL_FOUND})
                MESSAGE(FATAL_ERROR 
                        "OpenCL backend selected but OpenCL could not be found")
        ELSE()
                SET(AURA_BACKEND_LIBRARIES ${OpenCL_LIBRARIES} )
                SET(AURA_BACKEND_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
                SET(AURA_BACKEND_CUDA 0)
                SET(AURA_BACKEND_OPENCL 1)
                SET(AURA_BACKEND_METAL 0)
                ADD_DEFINITIONS(-DAURA_BACKEND_OPENCL)
        ENDIF()
ENDIF()

IF(${AURA_BACKEND} STREQUAL METAL)
        IF(NOT ${METAL_FOUND})
                MESSAGE(FATAL_ERROR "OpenCL backend selected but OpenCL could not be found")
                ELSE()
                SET(AURA_BACKEND_CUDA 0)
                SET(AURA_BACKEND_OPENCL 0)
                SET(AURA_BACKEND_METAL 1)
                ADD_DEFINITIONS(-DAURA_BACKEND_METAL)
        ENDIF()
ENDIF()
STRING(TOLOWER "${AURA_BACKEND}" AURA_BACKEND_LC)
MESSAGE(STATUS "Using ${AURA_BACKEND} backend")
]]

# Compiler and linker flags
SET(CMAKE_CXX_FLAGS "-Wall -std=c++0x -g -O3")
IF(APPLE)
        # Treat everything as Objective-C++.
        ADD_DEFINITIONS("-x objective-c++")
        SET(CMAKE_EXE_LINKER_FLAGS "-framework Foundation -w")
ENDIF()

# Set global include directories for project.
INCLUDE_DIRECTORIES("${AURA_BACKEND_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include/")
INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}/")

# Enable testing.
ENABLE_TESTING()

# Add subdirectories.
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/test/)
