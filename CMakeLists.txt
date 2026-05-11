cmake_minimum_required(VERSION 3.21)

project(cml VERSION 0.1.0 LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json" FORCE)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(CML_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" ON)
option(CML_FETCH_OPENBLAS "Fetch OpenBLAS when no system BLAS is found" ON)

include(CheckIncludeFile)
include(FetchContent)

set(BLA_STATIC ON)
find_package(BLAS QUIET)

set(CML_BLAS_TARGET "")
set(CML_CBLAS_INCLUDE_DIR "")
if(BLAS_FOUND)
    set(CML_BLAS_TARGET BLAS::BLAS)
    check_include_file(cblas.h CML_HAVE_CBLAS_H)
    if(NOT CML_HAVE_CBLAS_H)
        message(FATAL_ERROR "CBLAS header (cblas.h) not found. Install a BLAS provider with CBLAS support.")
    endif()
elseif(CML_FETCH_OPENBLAS)
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
        target_compile_options(${target} PRIVATE /W4 /WX)
    else()
        target_compile_options(
            ${target}
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                -Werror
                -Wcast-qual
                -Wconversion
                -Wdouble-promotion
                -Wformat=2
                -Wmissing-declarations
                -Wmissing-prototypes
                -Wnull-dereference
                -Wold-style-definition
                -Wredundant-decls
                -Wshadow
                -Wsign-conversion
                -Wstrict-prototypes
                -Wswitch-enum
                -Wundef
                -Wunreachable-code
                -Wwrite-strings
                -fno-common
        )
    endif()
endfunction()

function(cml_apply_sanitizers target)
    if(CML_ENABLE_SANITIZERS AND NOT MSVC)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

add_library(cml STATIC src/cml.c)
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

cml_apply_warnings(cml)
cml_apply_sanitizers(cml)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
