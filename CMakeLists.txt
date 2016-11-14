CMAKE_MINIMUM_REQUIRED(VERSION 2.8)


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
FIND_PACKAGE(Boost COMPONENTS system thread filesystem unit_test_framework regex python)


# Find GPU libraries.
FIND_PACKAGE(CUDA) 
FIND_PACKAGE(OpenCL)


# Find apple specific libraries.
IF (APPLE)
        FIND_LIBRARY(FOUNDATION_LIB Foundation)
        IF (NOT FOUNDATION_LIB)
                MESSAGE(FATAL_ERROR "Foundation not found.")
        ENDIF()
ENDIF()

# Set correct defines for GPU libraries. 
IF (${AURA_BASE} STREQUAL CUDA)
        FIND_CUDA_HELPER_LIBS(nvrtc)
        SET(AURA_BASE_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_nvrtc_LIBRARY})
        SET(AURA_BASE_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
        IF (APPLE)
                SET(AURA_BASE_LINK_DIRS "/usr/local/cuda/lib")
                SET(AURA_BASE_LIBRARIES ${AURA_BASE_LIBRARIES} 
                                        "cuda")
        ENDIF()
        SET(AURA_BASE_DEFINE "-DAURA_BASE_CUDA")
ELSEIF (${AURA_BASE} STREQUAL METAL)
        IF(NOT APPLE)
                MESSAGE(FATAL_ERROR "Metal only supported on Apple platform.")
        ENDIF()
        SET(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}
                                   "-framework Metal -w")
        SET(AURA_BASE_INCLUDE_DIRS ".")
        SET(AURA_BASE_DEFINE "-DAURA_BASE_METAL")
ELSEIF (${AURA_BASE} STREQUAL OPENCL)
        SET(AURA_BASE_LIBRARIES ${OPENCL_LIBRARIES})
        SET(AURA_BASE_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
        SET(AURA_BASE_DEFINE "-DAURA_BASE_OPENCL")
ELSE()
        MESSAGE(FATAL_ERROR "${AURA_BASE} is not a supported Aura base.")
ENDIF()
ADD_DEFINITIONS(${AURA_BASE_DEFINE})


# Compiler and linker flags
ADD_COMPILE_OPTIONS(-std=c++0x -g -O3)
IF (WIN32)
        ADD_COMPILE_OPTIONS(/EHsc)
        ADD_COMPILE_OPTIONS(/W3)
ELSE()
        # Prevent VS from spewing lots of warnings.
        ADD_COMPILE_OPTIONS(-Wall)
        ADD_COMPILE_OPTIONS(-Werror)
ENDIF()

# Set global include directories for project.
INCLUDE_DIRECTORIES("${AURA_BASE_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include/")
INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}/")


# Set global link directories for project.
LINK_DIRECTORIES(${LINK_DIRECTORIES} ${AURA_BASE_LINK_DIRS})


# Enable testing.
ENABLE_TESTING()

ADD_DEFINITIONS(-DAURA_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}/test")

# Add subdirectories.
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/test/)

# WIP
IF (${BUILD_PYTHON})
        ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/python/)
ENDIF()

# Documentation
IF (${BUILD_DOCUMENTATION})
        FIND_PACKAGE(Doxygen)
        IF (DOXYGEN_FOUND)
                # Build doxygen XML.
                CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in 
                        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
                ADD_CUSTOM_TARGET(doxygen 
                        ${DOXYGEN_EXECUTABLE} 
                        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        COMMENT "Generating API documentation with Doxygen" 
                        VERBATIM)
                # Run sphinx (with breathe).
                CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/doc/source/conf.py.in 
                        ${CMAKE_CURRENT_BINARY_DIR}/conf.py 
                        @ONLY)
                ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doc/html
                        COMMAND sphinx-build -b html -c . 
                                ${CMAKE_CURRENT_SOURCE_DIR}/doc/source 
                                ${CMAKE_CURRENT_BINARY_DIR}/doc/html
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        )
                # Make a target that builds all of the above.
                ADD_CUSTOM_TARGET(sphinx 
                        ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doc/html)
                ADD_CUSTOM_TARGET(doc)
                ADD_DEPENDENCIES(doc doxygen sphinx)
        ENDIF(DOXYGEN_FOUND)
ENDIF()
