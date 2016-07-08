
SET(PYTHON_INCLUDE_DIR "/Users/sschaetz/.pyenv/versions/3.4.3/include/python3.4m")
SET(PYTHON_LIBRARY "/Users/sschaetz/.pyenv/versions/3.4.3/lib")

FIND_PACKAGE(PythonLibs 3.4.3)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include/")

IF(APPLE)
        SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
ENDIF(APPLE)

ADD_LIBRARY(aura_opencl SHARED aura_opencl.cpp)
MESSAGE(${PYTHON_LIBRARIES})
TARGET_LINK_LIBRARIES(aura_opencl "/Users/sschaetz/.pyenv/versions/3.4.3/lib/libpython3.4m.dylib" ${Boost_LIBRARIES} ${AURA_BASE_LIBRARIES})
SET_TARGET_PROPERTIES(aura_opencl PROPERTIES PREFIX "")

