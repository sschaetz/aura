# AURA_ADD_TARGET
# creates an executable target with name TARGET_NAME from a number
# of source files and a number of libraries
# the first source file name has to be specified right after the library
# additional source files and libraries can be specified after the target name
FUNCTION(AURA_ADD_TARGET TARGET_NAME FIRST_SOURCE_FILENAME)
  SET(EXECUTABLE_SOURCE_FILES "${FIRST_SOURCE_FILENAME}")
  SET(EXECUTABLE_LINK_LIBRARIES "")
  SET(EXECUTABLE_DEPENDENCIES "")
  FOREACH(ARG ${ARGN})
    # check if source file, kernel file or library
    IF(${ARG} MATCHES "^[a-z/_]*.cpp$")
      SET(EXECUTABLE_SOURCE_FILES ${EXECUTABLE_SOURCE_FILES} ${ARG})
    ELSEIF(${ARG} MATCHES "^[a-z/_]*.cu$")
      IF(AURA_BACKEND_CUDA) 
        # Get name of CUDA module
        GET_FILENAME_COMPONENT(TMP ${ARG} NAME_WE)
        # FindCUDA command to build PTX 
        CUDA_COMPILE_PTX(SRC ${ARG})
        # Add a custom target to force PTX to be built
        ADD_CUSTOM_TARGET(${TARGET_NAME}_${TMP} SOURCES ${SRC})
        # This target must be added as dependency to the main test case
        SET(EXECUTABLE_DEPENDENCIES ${EXECUTABLE_DEPENDENCIES} ${TARGET_NAME}_${TMP})
        # Rename the file after it was built
        ADD_CUSTOM_COMMAND(TARGET ${TARGET_NAME}_${TMP} POST_BUILD COMMAND
          ${CMAKE_COMMAND} -E copy_if_different 
          ${SRC} ${CMAKE_CURRENT_BINARY_DIR}/${TMP}.ptx)
      ENDIF()
    ELSEIF(${ARG} MATCHES "^[a-z/_]*.cl$")
      IF(AURA_BACKEND_OPENCL) 
        # copy file
        FILE(COPY ${ARG} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
      ENDIF()
    ELSE()
      SET(EXECUTABLE_LINK_LIBRARIES ${EXECUTABLE_LINK_LIBRARIES} ${ARG})
    ENDIF()
  ENDFOREACH()
  # Debug messages:
  #MESSAGE("source files ${EXECUTABLE_SOURCE_FILES}")
  #MESSAGE("link files ${EXECUTABLE_LINK_LIBRARIES}")
  ADD_EXECUTABLE(${TARGET_NAME} ${EXECUTABLE_SOURCE_FILES})
  TARGET_LINK_LIBRARIES(${TARGET_NAME} ${EXECUTABLE_LINK_LIBRARIES})
  IF(NOT ${EXECUTABLE_DEPENDENCIES} STREQUAL "")
    #MESSAGE("Adding dependency ${TARGET_NAME} ${EXECUTABLE_DEPENDENCIES}")
    ADD_DEPENDENCIES(${TARGET_NAME} ${EXECUTABLE_DEPENDENCIES})
  ENDIF()
ENDFUNCTION()

# AURA_ADD_TEST
# creates a unit test, the name of the test ist derived from the
# folder and name of the first source file
FUNCTION(AURA_ADD_TEST FIRST_SOURCE_FILENAME)
  GET_FILENAME_COMPONENT(TARGET_NAME1 ${FIRST_SOURCE_FILENAME} PATH)
  GET_FILENAME_COMPONENT(TARGET_NAME2 ${FIRST_SOURCE_FILENAME} NAME_WE)
  SET(TARGET_NAME "test.${TARGET_NAME1}.${TARGET_NAME2}")
  SET(TEST_NAME "${TARGET_NAME1}.${TARGET_NAME2}")
  # Debug message:
  #MESSAGE("${TARGET_NAME} ${TEST_NAME}")
  AURA_ADD_TARGET(${TARGET_NAME} ${FIRST_SOURCE_FILENAME} ${ARGN})
  TARGET_LINK_LIBRARIES(${TARGET_NAME} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  ADD_TEST(${TEST_NAME} ${TARGET_NAME})
ENDFUNCTION()

