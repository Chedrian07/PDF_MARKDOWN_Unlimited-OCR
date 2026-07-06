cmake_minimum_required(VERSION 3.15...4.2)

project(uocr_native LANGUAGES CXX)

# C++17, Release/-O3 by default.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               Debug Release RelWithDebInfo MinSizeRel)
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(uocr_native src/uocr_native.cpp)

target_compile_features(uocr_native PRIVATE cxx_std_17)

# Explicit -O3 on optimized configs (GCC/Clang already imply it in Release,
# but the spec asks for it explicitly).
target_compile_options(uocr_native PRIVATE
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Release,RelWithDebInfo>>:-O3>)

# Install to the wheel root so the module is importable as `uocr_native`.
install(TARGETS uocr_native
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .)
