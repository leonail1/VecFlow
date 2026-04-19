# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(../../cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_cuvs)

project(test_cuvs LANGUAGES CXX CUDA)
get_filename_component(VECFLOW_CUDA_TOOLKIT_ROOT "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(VECFLOW_CUDA_TOOLKIT_ROOT "${VECFLOW_CUDA_TOOLKIT_ROOT}/.." ABSOLUTE)
set(CUDAToolkit_ROOT "${VECFLOW_CUDA_TOOLKIT_ROOT}" CACHE PATH "VecFlow examples CUDA toolkit root." FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "${VECFLOW_CUDA_TOOLKIT_ROOT}" CACHE PATH "Legacy FindCUDA toolkit root used by CUTLASS." FORCE)
set(CUDA_BIN_PATH "${VECFLOW_CUDA_TOOLKIT_ROOT}" CACHE PATH "Legacy FindCUDA toolkit root used by CUTLASS." FORCE)
set(CUDA_NVCC_EXECUTABLE "${CMAKE_CUDA_COMPILER}" CACHE FILEPATH "Legacy FindCUDA nvcc path used by CUTLASS." FORCE)
find_package(CUDAToolkit REQUIRED)
find_package(Threads)
set(VECFLOW_TAGORE_DIM_SIZE
    192
    CACHE STRING "Compile-time Tagore dimension upper bound used by the VecFlow example targets.")

option(VECFLOW_BENCH_PROGRESS_LOG
  "Enable lightweight progress timestamps in VECFLOW_BENCH (zero cost when OFF)."
  OFF)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")
endif()

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY OFF)
include(../../cmake/thirdparty/get_cuvs.cmake)

# ------------- configure nlohmann/json -----------------#

rapids_cpm_find(nlohmann_json
  VERSION 3.11.2
  GITHUB_REPOSITORY nlohmann/json
  OPTIONS
    "JSON_BuildTests OFF"
    "CMAKE_POLICY_VERSION_MINIMUM 3.5"
)

# -------------- compile tasks ----------------- #
add_executable(VECFLOW_EXAMPLE src/vecflow_example.cu)
add_executable(VECFLOW_BENCH src/bench/vecflow_bench.cu)
add_executable(VECFLOW_QPS_TIMESERIES src/bench/vecflow_qps_timeseries.cu)
add_executable(VECFLOW_MG_BENCH src/bench/vecflow_mg_bench.cpp)
add_executable(VECFLOW_TAGORE_SMOKE src/bench/vecflow_tagore_smoke.cu)
add_executable(VECFLOW_MG_API_SMOKE src/bench/vecflow_mg_api_smoke.cu)

set(VECFLOW_LOCAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(VECFLOW_LOCAL_VECFLOW_SOURCES
  ${VECFLOW_LOCAL_ROOT}/cpp/src/neighbors/cagra_filtered_search_float.cu
  ${VECFLOW_LOCAL_ROOT}/cpp/src/neighbors/detail/cagra/search_multi_cta_float_uint32.cu
  ${VECFLOW_LOCAL_ROOT}/cpp/src/neighbors/detail/cagra/filtered_search_single_cta_float_uint32.cu
  ${VECFLOW_LOCAL_ROOT}/cpp/src/neighbors/filtered_bfs/detail/build_filtered_bfs_float.cu
  ${VECFLOW_LOCAL_ROOT}/cpp/src/neighbors/filtered_bfs/detail/search_filtered_bfs_float.cu
  ${VECFLOW_LOCAL_ROOT}/cpp/src/neighbors/vecflow/detail/build_vecflow_float.cu
  ${VECFLOW_LOCAL_ROOT}/cpp/src/neighbors/vecflow/detail/search_vecflow_float.cu
  ${VECFLOW_LOCAL_ROOT}/thirdparty/Tagore/src/Tagore_src.cu
)

function(vecflow_configure_local_cuda_target target_name)
  target_sources(${target_name} PRIVATE ${VECFLOW_LOCAL_VECFLOW_SOURCES})
  target_include_directories(${target_name} PRIVATE
    ${VECFLOW_LOCAL_ROOT}/cpp/include
    ${VECFLOW_LOCAL_ROOT}/thirdparty/Tagore/include
    /usr/include
  )
  target_compile_definitions(${target_name} PRIVATE
    CUVS_VECFLOW_TAGORE_ENABLED
    DIM_SIZE=${VECFLOW_TAGORE_DIM_SIZE}
    FMT_SHARED
  )
  # These targets compile local VecFlow CUDA sources directly. Force shared fmt
  # so nvcc 12.4 does not inherit the transitive header-only configuration.
  target_compile_options(${target_name} PRIVATE -UFMT_HEADER_ONLY)
  target_link_libraries(${target_name} PRIVATE fmt::fmt)
  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endfunction()

vecflow_configure_local_cuda_target(VECFLOW_EXAMPLE)
vecflow_configure_local_cuda_target(VECFLOW_BENCH)
vecflow_configure_local_cuda_target(VECFLOW_QPS_TIMESERIES)
vecflow_configure_local_cuda_target(VECFLOW_TAGORE_SMOKE)
vecflow_configure_local_cuda_target(VECFLOW_MG_API_SMOKE)

# `$<TARGET_NAME_IF_EXISTS:conda_env>` is a generator expression that ensures that targets are
# installed in a conda environment, if one exists
target_link_libraries(VECFLOW_EXAMPLE PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE nlohmann_json::nlohmann_json PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(VECFLOW_BENCH PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE nlohmann_json::nlohmann_json PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
if(VECFLOW_BENCH_PROGRESS_LOG)
  target_compile_definitions(VECFLOW_BENCH PRIVATE VECFLOW_BENCH_PROGRESS_LOG)
endif()
target_link_libraries(VECFLOW_QPS_TIMESERIES PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE nlohmann_json::nlohmann_json PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(VECFLOW_MG_BENCH PRIVATE nlohmann_json::nlohmann_json PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(VECFLOW_TAGORE_SMOKE PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(VECFLOW_MG_API_SMOKE PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)

set(VECFLOW_PHOENIX_BUILD_DIR "")
set(VECFLOW_PHOENIX_CANDIDATES
  "${VECFLOW_LOCAL_ROOT}/thirdparty/phoenix/build-a100"
  "${VECFLOW_LOCAL_ROOT}/thirdparty/phoenix/build"
  "${VECFLOW_LOCAL_ROOT}/thirdparty/phoenix/build-dev"
)
foreach(_phoenix_dir IN LISTS VECFLOW_PHOENIX_CANDIDATES)
  if(EXISTS "${_phoenix_dir}/libphoenix.so")
    set(VECFLOW_PHOENIX_BUILD_DIR "${_phoenix_dir}")
    break()
  endif()
endforeach()
set(VECFLOW_PHOENIX_LIB "${VECFLOW_PHOENIX_BUILD_DIR}/libphoenix.so")
set(VECFLOW_PHOENIX_INCLUDE "${VECFLOW_LOCAL_ROOT}/thirdparty/phoenix/libphoenix/include")

if(EXISTS "${VECFLOW_PHOENIX_LIB}")
  if(NOT TARGET vecflow_phoenix)
    add_library(vecflow_phoenix SHARED IMPORTED GLOBAL)
    set_target_properties(vecflow_phoenix PROPERTIES
      IMPORTED_LOCATION "${VECFLOW_PHOENIX_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${VECFLOW_PHOENIX_INCLUDE}"
    )
  endif()

  target_link_libraries(VECFLOW_EXAMPLE PRIVATE vecflow_phoenix)
  target_link_libraries(VECFLOW_BENCH PRIVATE vecflow_phoenix)
  target_link_libraries(VECFLOW_QPS_TIMESERIES PRIVATE vecflow_phoenix)
  target_link_libraries(VECFLOW_MG_API_SMOKE PRIVATE vecflow_phoenix)
  target_compile_definitions(VECFLOW_EXAMPLE PRIVATE CUVS_VECFLOW_PHOENIX_ENABLED)
  target_compile_definitions(VECFLOW_BENCH PRIVATE CUVS_VECFLOW_PHOENIX_ENABLED)
  target_compile_definitions(VECFLOW_QPS_TIMESERIES PRIVATE CUVS_VECFLOW_PHOENIX_ENABLED)
  target_compile_definitions(VECFLOW_MG_API_SMOKE PRIVATE CUVS_VECFLOW_PHOENIX_ENABLED)
  set_property(TARGET VECFLOW_EXAMPLE APPEND PROPERTY BUILD_RPATH "${VECFLOW_PHOENIX_BUILD_DIR}")
  set_property(TARGET VECFLOW_BENCH APPEND PROPERTY BUILD_RPATH "${VECFLOW_PHOENIX_BUILD_DIR}")
  set_property(TARGET VECFLOW_QPS_TIMESERIES APPEND PROPERTY BUILD_RPATH "${VECFLOW_PHOENIX_BUILD_DIR}")
  set_property(TARGET VECFLOW_MG_API_SMOKE APPEND PROPERTY BUILD_RPATH "${VECFLOW_PHOENIX_BUILD_DIR}")

  add_executable(VECFLOW_PHOENIX_SMOKE src/bench/vecflow_phoenix_smoke.cpp)
  target_link_libraries(VECFLOW_PHOENIX_SMOKE PRIVATE
    vecflow_phoenix
    CUDA::cudart
    $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  set_target_properties(VECFLOW_PHOENIX_SMOKE PROPERTIES
    BUILD_RPATH "${VECFLOW_PHOENIX_BUILD_DIR}"
  )
else()
  message(STATUS "Skipping VECFLOW_PHOENIX_SMOKE because libphoenix.so was not found")
endif()
