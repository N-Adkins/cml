cmake_minimum_required(VERSION 3.21)

project(cml VERSION 0.1.0 LANGUAGES C)

include(CTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json" FORCE)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(CML_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" ON)
option(CML_FETCH_OPENBLAS "Fetch OpenBLAS when no system BLAS is found" ON)
option(CML_ENABLE_CUDA "Enable CUDA backend support" OFF)
option(CML_CUDA_ALLOW_UNSUPPORTED_COMPILER
    "Pass -allow-unsupported-compiler to nvcc (useful with newer GCC versions)"
    ON
)

if(CML_ENABLE_CUDA)
    set(_cml_cuda_standard 14)
    if(CML_CUDA_ALLOW_UNSUPPORTED_COMPILER)
        set(_cml_cuda_standard 20)
        if(NOT CMAKE_CUDA_FLAGS MATCHES "(^| )-allow-unsupported-compiler( |$)")
            string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
        endif()
        if(NOT CMAKE_CUDA_FLAGS MATCHES "(^| )-std=c\\+\\+20( |$)")
            string(APPEND CMAKE_CUDA_FLAGS " -std=c++20")
        endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD ${_cml_cuda_standard})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
endif()

include(FetchContent)

find_package(BLAS QUIET)

set(CML_BLAS_TARGET "")
if(BLAS_FOUND)
    set(CML_BLAS_TARGET BLAS::BLAS)

    # Derive candidate include dirs from the BLAS library path (e.g. /usr/lib64 -> /usr/include)
    list(GET BLAS_LIBRARIES 0 _cml_blas_lib)
    get_filename_component(_cml_blas_lib_dir "${_cml_blas_lib}" DIRECTORY)

    # Clear any stale cached result before searching
    unset(CML_CBLAS_INCLUDE_DIR)
    unset(CML_CBLAS_INCLUDE_DIR CACHE)

    # Search adjacent to the BLAS library first (handles non-default prefixes)
    find_path(CML_CBLAS_INCLUDE_DIR
        NAMES cblas.h
        PATHS
            "${_cml_blas_lib_dir}/../include/openblas"
            "${_cml_blas_lib_dir}/../include/atlas"
            "${_cml_blas_lib_dir}/../include/blas"
            "${_cml_blas_lib_dir}/../include"
        NO_DEFAULT_PATH
    )

    # Fall back to system-wide search
    if(NOT CML_CBLAS_INCLUDE_DIR)
        unset(CML_CBLAS_INCLUDE_DIR CACHE)
        find_path(CML_CBLAS_INCLUDE_DIR
            NAMES cblas.h
            PATH_SUFFIXES openblas atlas blas ""
        )
    endif()

    if(NOT CML_CBLAS_INCLUDE_DIR)
        message(FATAL_ERROR
            "System BLAS was found (${BLAS_LIBRARIES}) but cblas.h was not found. "
            "Install the BLAS development headers (e.g. openblas-devel).")
    endif()
elseif(CML_FETCH_OPENBLAS)
    set(PRIOR_BUILD_TESTING ${BUILD_TESTING})

    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(NOFORTRAN ON CACHE BOOL "" FORCE)
    set(ONLY_CBLAS ON CACHE BOOL "" FORCE)
    set(DYNAMIC_ARCH OFF CACHE BOOL "" FORCE)
    set(TARGET GENERIC CACHE STRING "" FORCE)

    FetchContent_Declare(
        openblas
        URL https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.30.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(openblas)

    set(BUILD_TESTING ${PRIOR_BUILD_TESTING} CACHE BOOL "" FORCE)

    if(TARGET openblas)
        set(CML_BLAS_TARGET openblas)
        set(CML_CBLAS_INCLUDE_DIR "${openblas_SOURCE_DIR};${openblas_BINARY_DIR}")
    else()
        message(FATAL_ERROR "Failed to fetch/build OpenBLAS target.")
    endif()
else()
    message(FATAL_ERROR "No BLAS provider found. Install one or set -DCML_FETCH_OPENBLAS=ON.")
endif()

function(cml_apply_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:/W4>
            $<$<COMPILE_LANGUAGE:C>:/WX>
        )
    else()
        target_compile_options(
            ${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:C>:-Wall>
                $<$<COMPILE_LANGUAGE:C>:-Wextra>
                $<$<COMPILE_LANGUAGE:C>:-Wpedantic>
                $<$<COMPILE_LANGUAGE:C>:-Werror>
                $<$<COMPILE_LANGUAGE:C>:-Wcast-qual>
                $<$<COMPILE_LANGUAGE:C>:-Wconversion>
                $<$<COMPILE_LANGUAGE:C>:-Wdouble-promotion>
                $<$<COMPILE_LANGUAGE:C>:-Wformat=2>
                $<$<COMPILE_LANGUAGE:C>:-Wmissing-declarations>
                $<$<COMPILE_LANGUAGE:C>:-Wmissing-prototypes>
                $<$<COMPILE_LANGUAGE:C>:-Wnull-dereference>
                $<$<COMPILE_LANGUAGE:C>:-Wold-style-definition>
                $<$<COMPILE_LANGUAGE:C>:-Wredundant-decls>
                $<$<COMPILE_LANGUAGE:C>:-Wshadow>
                $<$<COMPILE_LANGUAGE:C>:-Wsign-conversion>
                $<$<COMPILE_LANGUAGE:C>:-Wstrict-prototypes>
                $<$<COMPILE_LANGUAGE:C>:-Wswitch-enum>
                $<$<COMPILE_LANGUAGE:C>:-Wundef>
                $<$<COMPILE_LANGUAGE:C>:-Wunreachable-code>
                $<$<COMPILE_LANGUAGE:C>:-Wwrite-strings>
                $<$<COMPILE_LANGUAGE:C>:-fno-common>
        )
    endif()
endfunction()

function(cml_apply_sanitizers target)
    if(CML_ENABLE_SANITIZERS AND NOT MSVC AND NOT CML_ENABLE_CUDA)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

# Stage CUDA DLLs next to exes on Windows; FindCUDAToolkit doesn't set
# IMPORTED_LOCATION on its imported targets so $<TARGET_RUNTIME_DLLS> is empty
set(CML_RUNTIME_DLLS "")
if(WIN32 AND CML_ENABLE_CUDA)
    file(GLOB CML_RUNTIME_DLLS
        "${CUDAToolkit_BIN_DIR}/cudart64_*.dll"
        "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
        "${CUDAToolkit_BIN_DIR}/cublasLt64_*.dll"
        "${CUDAToolkit_BIN_DIR}/nvJitLink64_*.dll"
    )
endif()

function(cml_stage_runtime_dlls target)
    if(NOT CML_RUNTIME_DLLS)
        return()
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CML_RUNTIME_DLLS} $<TARGET_FILE_DIR:${target}>
    )
endfunction()

add_library(cml STATIC)
add_library(cml::cml ALIAS cml)

target_include_directories(
    cml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(cml PUBLIC c_std_99)
if(CML_CBLAS_INCLUDE_DIR)
    target_include_directories(cml SYSTEM PRIVATE ${CML_CBLAS_INCLUDE_DIR})
endif()

target_link_libraries(cml PUBLIC ${CML_BLAS_TARGET})
if (NOT WIN32)
    target_link_libraries(cml PRIVATE m)
endif()
if(CML_ENABLE_CUDA)
    target_compile_definitions(cml PRIVATE CML_ENABLE_CUDA)
    target_link_libraries(cml PRIVATE CUDA::cudart CUDA::cublas)
endif()

cml_apply_warnings(cml)
cml_apply_sanitizers(cml)

add_subdirectory(src)

# Examples are added before tests because the integration test suite links the
# `cml_scenarios` library defined here. The runnable example binaries
# themselves are still gated by CML_BUILD_EXAMPLES.
option(CML_BUILD_EXAMPLES "Build example programs" ON)
add_subdirectory(examples)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
