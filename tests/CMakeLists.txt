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
    cml_stage_runtime_dlls(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

add_unity_test(test_context.c)
add_unity_test(test_linear.c)
add_unity_test(test_loss.c)
add_unity_test(test_nn.c)
add_unity_test(test_sgd.c)
add_unity_test(test_tape.c)
add_unity_test(test_tensor.c)
add_unity_test(test_trainer.c)

# Integration suite. Always built; CUDA cases self-skip when no device is
# available at runtime (cml_cuda_is_available() guards their registration).
add_executable(test_integration test_integration.c)
target_link_libraries(test_integration PRIVATE cml ${_unity_target})
target_include_directories(test_integration PRIVATE ../src ../include)

cml_apply_warnings(test_integration)
cml_apply_sanitizers(test_integration)
cml_stage_runtime_dlls(test_integration)

# Integration scenarios train for thousands of epochs under sanitizers; give
# the runner a longer budget than the per-op unit tests.
add_test(NAME test_integration COMMAND test_integration)
set_tests_properties(test_integration PROPERTIES TIMEOUT 300)

if(CML_ENABLE_CUDA)
    add_unity_test(test_cuda_parity.c)
endif()
