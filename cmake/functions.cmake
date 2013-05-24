# AURA_ADD_TARGET
# creates an executable target with name TARGET_NAME from a number
# of source files and a number of libraries
# the first source file name has to be specified right after the library
# additional source files and libraries can be specified after the target name
FUNCTION(AURA_ADD_TARGET TARGET_NAME FIRST_SOURCE_FILENAME)
  SET(EXECUTABLE_SOURCE_FILES "${FIRST_SOURCE_FILENAME}")
  SET(EXECUTABLE_LINK_LIBRARIES "")
  FOREACH(ARG ${ARGN})
    # check if source file or library
    IF(${ARG} MATCHES "([a-z_]*.cpp)|[a-z_]*.cu|[a-z_]*.cl")
      SET(EXECUTABLE_SOURCE_FILES ${ARG} ${EXECUTABLE_SOURCE_FILES})
    ELSE()
      SET(EXECUTABLE_LINK_LIBRARIES ${ARG} ${EXECUTABLE_LINK_LIBRARIES})
    ENDIF()
  ENDFOREACH()
  #MESSAGE("${EXECUTABLE_SOURCE_FILES} ${EXECUTABLE_LINK_LIBRARIES}")
  ADD_EXECUTABLE(${TARGET_NAME} ${EXECUTABLE_SOURCE_FILES})
  TARGET_LINK_LIBRARIES(${TARGET_NAME} ${EXECUTABLE_LINK_LIBRARIES})
ENDFUNCTION()

