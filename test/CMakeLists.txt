# Helper function to define tests.
FUNCTION(ADD_AURA_TEST TEST_NAME)
        ADD_EXECUTABLE(${TEST_NAME} ${ARGN})
        TARGET_LINK_LIBRARIES(${TEST_NAME}
                              ${AURA_BASE_LIBRARIES}
                              ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
	                      ${Boost_SYSTEM_LIBRARY}
                              ${Boost_REGEX_LIBRARY})
        ADD_TEST(${TEST_NAME} ${TEST_NAME})
        FOREACH(TEST_SOURCE ${ARGN})
                IF (APPLE)
                        SET_SOURCE_FILES_PROPERTIES(${TEST_SOURCE}
                                PROPERTIES COMPILE_FLAGS "-x objective-c++")
                ENDIF()
        ENDFOREACH()
        TARGET_LINK_LIBRARIES(${TEST_NAME} ${FOUNDATION_LIB})
        TARGET_INCLUDE_DIRECTORIES(${TEST_NAME} PUBLIC
                "${PROJECT_SOURCE_DIR}/")
ENDFUNCTION()

ADD_AURA_TEST(test.alang alang.cpp alang.cpp)
ADD_AURA_TEST(test.copy copy.cpp)
ADD_AURA_TEST(test.device device.cpp)
ADD_AURA_TEST(test.device_array device_array.cpp)
ADD_AURA_TEST(test.device_ptr device_ptr.cpp)
ADD_AURA_TEST(test.feed feed.cpp)
ADD_AURA_TEST(test.invoke invoke.cpp)
ADD_AURA_TEST(test.io io.cpp)
ADD_AURA_TEST(test.library library.cpp)
ADD_AURA_TEST(test.multi_comp_units multi_comp_units1.cpp multi_comp_units2.cpp)
ADD_AURA_TEST(test.preprocessor preprocessor.cpp)
ADD_AURA_TEST(test.tiny_vector tiny_vector.cpp)

