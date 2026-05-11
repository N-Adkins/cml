include(FetchContent)

FetchContent_Declare(
    unity
    URL https://github.com/ThrowTheSwitch/Unity/archive/refs/tags/v2.6.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(unity)

if(TARGET unity)
    set(_unity_target unity)
elseif(TARGET Unity)
    set(_unity_target Unity)
else()
    add_library(unity STATIC ${unity_SOURCE_DIR}/src/unity.c)
    target_include_directories(unity PUBLIC ${unity_SOURCE_DIR}/src)
    set(_unity_target unity)
endif()

add_executable(cml_tests test_cml.c)
target_link_libraries(cml_tests PRIVATE cml ${_unity_target})

cml_apply_warnings(cml_tests)
cml_apply_sanitizers(cml_tests)

add_test(NAME cml_tests COMMAND cml_tests)
