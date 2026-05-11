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

function(get_filename_stem input_file output_var)
    get_filename_component(temp_stem "${input_file}" NAME_WE)
    set(${output_var} "${temp_stem}" PARENT_SCOPE)
endfunction()

macro(add_unity_test file_name)
    get_filename_stem("${file_name}" test_name)
    add_executable(${test_name} ${file_name})
    target_link_libraries(${test_name} PRIVATE cml ${_unity_target})
    target_include_directories(${test_name} PRIVATE ../src ../include)
    cml_apply_warnings(${test_name})
    cml_apply_sanitizers(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

add_unity_test(test_context.c)
add_unity_test(test_tensor.c)
add_unity_test(test_tape.c)
