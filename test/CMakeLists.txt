# Helper function to define tests.
FUNCTION(ADD_AURA_TEST TEST_NAME TEST_SOURCE)
        ADD_EXECUTABLE(${TEST_NAME} ${TEST_SOURCE})
        TARGET_LINK_LIBRARIES(${TEST_NAME} 
                              ${AURA_BACKEND_LIBRARIES} 
                              ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} 
	                      ${Boost_SYSTEM_LIBRARY})
        ADD_TEST(${TEST_NAME} ${TEST_NAME})
ENDFUNCTION()

ADD_AURA_TEST(test.device device.cpp)

