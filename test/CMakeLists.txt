# Helper function to define tests.
FUNCTION(ADD_AURA_TEST TEST_NAME TEST_SOURCE)
        ADD_EXECUTABLE(${TEST_NAME} ${TEST_SOURCE})
        TARGET_LINK_LIBRARIES(${TEST_NAME}
                              ${AURA_BASE_LIBRARIES}
                              ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
	                      ${Boost_SYSTEM_LIBRARY})
        ADD_TEST(${TEST_NAME} ${TEST_NAME})
        IF(APPLE)
                SET_SOURCE_FILES_PROPERTIES(${TEST_SOURCE}
                        PROPERTIES COMPILE_FLAGS "-x objective-c++")
        ENDIF()
ENDFUNCTION()

ADD_AURA_TEST(test.device device.cpp)

