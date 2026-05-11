cmake_minimum_required(VERSION 3.21)

project(cml VERSION 0.1.0 LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json" FORCE)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(CML_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" ON)

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

cml_apply_warnings(cml)
cml_apply_sanitizers(cml)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
