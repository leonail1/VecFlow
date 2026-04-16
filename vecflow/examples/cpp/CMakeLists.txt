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
find_package(Threads)
set(VECFLOW_TAGORE_DIM_SIZE
    192
    CACHE STRING "Compile-time Tagore dimension upper bound used by the VecFlow example targets.")

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

target_sources(VECFLOW_EXAMPLE PRIVATE ${VECFLOW_LOCAL_VECFLOW_SOURCES})
target_sources(VECFLOW_BENCH PRIVATE ${VECFLOW_LOCAL_VECFLOW_SOURCES})
target_sources(VECFLOW_TAGORE_SMOKE PRIVATE ${VECFLOW_LOCAL_VECFLOW_SOURCES})
target_sources(VECFLOW_MG_API_SMOKE PRIVATE ${VECFLOW_LOCAL_VECFLOW_SOURCES})

target_include_directories(VECFLOW_EXAMPLE PRIVATE
  ${VECFLOW_LOCAL_ROOT}/cpp/include
  ${VECFLOW_LOCAL_ROOT}/thirdparty/Tagore/include
  /usr/include
)
target_include_directories(VECFLOW_BENCH PRIVATE
  ${VECFLOW_LOCAL_ROOT}/cpp/include
  ${VECFLOW_LOCAL_ROOT}/thirdparty/Tagore/include
  /usr/include
)
target_include_directories(VECFLOW_TAGORE_SMOKE PRIVATE
  ${VECFLOW_LOCAL_ROOT}/cpp/include
  ${VECFLOW_LOCAL_ROOT}/thirdparty/Tagore/include
  /usr/include
)
target_include_directories(VECFLOW_MG_API_SMOKE PRIVATE
  ${VECFLOW_LOCAL_ROOT}/cpp/include
  ${VECFLOW_LOCAL_ROOT}/thirdparty/Tagore/include
  /usr/include
)

target_compile_definitions(VECFLOW_EXAMPLE PRIVATE
  CUVS_VECFLOW_TAGORE_ENABLED
  DIM_SIZE=${VECFLOW_TAGORE_DIM_SIZE}
)
target_compile_definitions(VECFLOW_BENCH PRIVATE
  CUVS_VECFLOW_TAGORE_ENABLED
  DIM_SIZE=${VECFLOW_TAGORE_DIM_SIZE}
)
target_compile_definitions(VECFLOW_TAGORE_SMOKE PRIVATE
  CUVS_VECFLOW_TAGORE_ENABLED
  DIM_SIZE=${VECFLOW_TAGORE_DIM_SIZE}
)
target_compile_definitions(VECFLOW_MG_API_SMOKE PRIVATE
  CUVS_VECFLOW_TAGORE_ENABLED
  DIM_SIZE=${VECFLOW_TAGORE_DIM_SIZE}
)

set_target_properties(VECFLOW_EXAMPLE PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(VECFLOW_BENCH PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(VECFLOW_TAGORE_SMOKE PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_target_properties(VECFLOW_MG_API_SMOKE PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# `$<TARGET_NAME_IF_EXISTS:conda_env>` is a generator expression that ensures that targets are
# installed in a conda environment, if one exists
target_link_libraries(VECFLOW_EXAMPLE PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE nlohmann_json::nlohmann_json PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(VECFLOW_BENCH PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE nlohmann_json::nlohmann_json PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
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
  target_link_libraries(VECFLOW_MG_API_SMOKE PRIVATE vecflow_phoenix)
  target_compile_definitions(VECFLOW_EXAMPLE PRIVATE CUVS_VECFLOW_PHOENIX_ENABLED)
  target_compile_definitions(VECFLOW_BENCH PRIVATE CUVS_VECFLOW_PHOENIX_ENABLED)
  target_compile_definitions(VECFLOW_MG_API_SMOKE PRIVATE CUVS_VECFLOW_PHOENIX_ENABLED)
  set_property(TARGET VECFLOW_EXAMPLE APPEND PROPERTY BUILD_RPATH "${VECFLOW_PHOENIX_BUILD_DIR}")
  set_property(TARGET VECFLOW_BENCH APPEND PROPERTY BUILD_RPATH "${VECFLOW_PHOENIX_BUILD_DIR}")
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
