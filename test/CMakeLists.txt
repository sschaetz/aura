# Helper function to define tests.
FUNCTION(ADD_AURA_TEST TEST_NAME TEST_SOURCE)
        ADD_EXECUTABLE(${TEST_NAME} ${TEST_SOURCE})
        TARGET_LINK_LIBRARIES(${TEST_NAME}
                              ${AURA_BASE_LIBRARIES}
                              ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
	                      ${Boost_SYSTEM_LIBRARY}
                              ${Boost_REGEX_LIBRARY})
        ADD_TEST(${TEST_NAME} ${TEST_NAME})
        IF (APPLE)
                SET_SOURCE_FILES_PROPERTIES(${TEST_SOURCE}
                        PROPERTIES COMPILE_FLAGS "-x objective-c++")
                TARGET_LINK_LIBRARIES(${TEST_NAME}
                                      ${FOUNDATION_LIB})
        ENDIF()
        TARGET_INCLUDE_DIRECTORIES(${TEST_NAME} PUBLIC
                "${PROJECT_SOURCE_DIR}/")
ENDFUNCTION()

ADD_AURA_TEST(test.invoke invoke.cpp)
ADD_AURA_TEST(test.basic basic.cpp)
ADD_AURA_TEST(test.copy copy.cpp)
ADD_AURA_TEST(test.preprocessor preprocessor.cpp)
ADD_AURA_TEST(test.device_ptr device_ptr.cpp)
ADD_AURA_TEST(test.io io.cpp)
ADD_AURA_TEST(test.library library.cpp)

