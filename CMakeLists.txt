CMAKE_MINIMUM_REQUIRED(VERSION 3.1)


# If no backend is selected, choose OpenCL as default. 
IF(NOT AURA_BASE)
        SET(AURA_BASE OPENCL)
ENDIF()


# Set unit test device id.
IF(NOT AURA_UNIT_TEST_DEVICE)
        SET(AURA_UNIT_TEST_DEVICE 0)
ENDIF()
ADD_DEFINITIONS(-DAURA_UNIT_TEST_DEVICE=${AURA_UNIT_TEST_DEVICE})


# Make additional modules available and source custom functions.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/buildprocess/")


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


# Find GPU libraries.
FIND_PACKAGE(CUDA) 
FIND_PACKAGE(OpenCL)


# Set correct defines for GPU libraries. 
IF(${AURA_BASE} STREQUAL CUDA)
        SET(AURA_BASE_LIBRARIES ${CUDA_LIBRARIES})
        SET(AURA_BASE_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
        IF(APPLE)
                SET(AURA_BASE_LINK_DIRS "/usr/local/cuda/lib")
                SET(AURA_BASE_LIBRARIES ${AURA_BASE_LIBRARIES} 
                                        "cuda")
        ENDIF()
        SET(AURA_BASE_DEFINE "-DAURA_BASE_CUDA")
ELSEIF(${AURA_BASE} STREQUAL METAL)
        IF(NOT APPLE)
                MESSAGE(FATAL_ERROR "Metal only supported on Apple platform.")
        ENDIF()
        SET(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}
                                   "-framework Metal -w")
        SET(AURA_BASE_INCLUDE_DIRS ".")
        SET(AURA_BASE_DEFINE "-DAURA_BASE_METAL")
ENDIF()

ADD_DEFINITIONS(${AURA_BASE_DEFINE})


# Compiler and linker flags
SET(CMAKE_CXX_FLAGS "-Wall -std=c++0x -g -O3")
IF(APPLE)
        # Treat everything as Objective-C++.
        #SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}
        #                            -framework Foundation -w")
ENDIF()


# Set global include directories for project.
INCLUDE_DIRECTORIES("${AURA_BASE_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include/")
INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}/")


# Set global link directories for project.
LINK_DIRECTORIES(${LINK_DIRECTORIES} ${AURA_BASE_LINK_DIRS})


# Enable testing.
ENABLE_TESTING()


# Add subdirectories.
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/test/)

