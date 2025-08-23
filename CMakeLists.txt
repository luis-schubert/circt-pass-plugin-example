cmake_minimum_required(VERSION 3.20.0)

project(CustomOperationPass LANGUAGES CXX C)

set(CMAKE_CXX_STANDÁRD 17 CACHE STRING "C++ standard")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Set LLVM/MLIR directory paths manually if not installed system-wide
# These should point to your LLVM build, not source
# Currently assumes, that circt is located adjacent to this project folder
set(LLVM_DIR "${CMAKE_SOURCE_DIR}/circt/llvm/build/lib/cmake/llvm")
set(MLIR_DIR "${CMAKE_SOURCE_DIR}/circt/llvm/build/lib/cmake/mlir")
set(CIRCT_DIR "${CMAKE_SOURCE_DIR}/circt/build/lib/cmake/circt")

# Find LLVM and MLIR packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM dir: ${LLVM_DIR}")
message(STATUS "MLIR dir: ${MLIR_DIR}")
message(STATUS "CIRCT dir: ${CIRCT_DIR}")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Needed for MLIR CMake macros
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
# Needed for MLIR CMake macros
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)

include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)
include(${CIRCT_CMAKE_DIR}/AddCIRCT.cmake)
include(HandleLLVMOptions)

set(CUSTOMOPERATION_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(CUSTOMOPERATION_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(
  SYSTEM ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CIRCT_INCLUDE_DIRS}
  ${CUSTOMOPERATION_SOURCE_DIR}/include
  ${CUSTOMOPERATION_BINARY_DIR}/include
)
link_directories(${LLVM_BUILD_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)

# Build your pass as a shared library (plugin)
add_llvm_library(CustomOperation
  MODULE
  ${CUSTOMOPERATION_SOURCE_DIR}/src/CustomOperation.cpp

  DEPENDS
  MLIRCustomOperationIncGen

  LINK_LIBS
  MLIRIR
  MLIRRegisterAllPasses
  CIRCTSupport
  # TODO add additional dependencies here
)

set_target_properties(CustomOperation PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

# Ensure we use the correct LLVM flags
llvm_update_compile_flags(CustomOperation)
