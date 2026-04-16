/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/device_vector.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <omp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip> 
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unistd.h>

#include "vecflow_common.cuh"
#include "multi_gpu.cuh"
#include "phoenix_graph_load.cuh"
#include "tagore_build.cuh"

namespace cuvs::neighbors::vecflow {

namespace detail {

template <typename data_t>
__global__ void gather_rows_for_cache_kernel(const data_t* source,
                                             const uint32_t* row_ids,
                                             data_t* destination,
                                             int64_t n_rows,
                                             int64_t dim)
{
  auto tid   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = n_rows * dim;
  if (tid >= total) { return; }

  auto row = tid / dim;
  auto col = tid % dim;
  destination[tid] = source[static_cast<int64_t>(row_ids[row]) * dim + col];
}

template <typename data_t>
void write_packed_dataset_cache(shared_resources::configured_raft_resources& res,
                                raft::device_matrix_view<const data_t, int64_t> dataset,
                                const std::vector<uint32_t>& index_map,
                                const std::string& filename)
{
  auto rows = static_cast<int64_t>(index_map.size());
  auto cols = static_cast<int64_t>(dataset.extent(1));
  auto file = ::open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (file < 0) { throw std::runtime_error("Cannot create dataset cache file: " + filename); }

  auto close_file = [&]() {
    if (file >= 0) {
      ::close(file);
      file = -1;
    }
  };

  try {
    if (::write(file, &rows, sizeof(rows)) != sizeof(rows) ||
        ::write(file, &cols, sizeof(cols)) != sizeof(cols)) {
      throw std::runtime_error("Cannot write dataset cache header: " + filename);
    }

    auto stream = raft::resource::get_cuda_stream(res);
    auto target_chunk_bytes = std::size_t{64} << 20;
    auto values_per_chunk =
      std::max<int64_t>(1, static_cast<int64_t>(target_chunk_bytes / sizeof(data_t)));
    auto chunk_rows = std::max<int64_t>(1, values_per_chunk / std::max<int64_t>(cols, 1));
    auto d_row_ids = raft::make_device_vector<uint32_t, int64_t>(res, chunk_rows);
    auto d_rows = raft::make_device_matrix<data_t, int64_t>(res, chunk_rows, cols);
    std::vector<data_t> host_rows(static_cast<std::size_t>(chunk_rows * cols));

    for (int64_t row_offset = 0; row_offset < rows; row_offset += chunk_rows) {
      auto rows_to_write = std::min<int64_t>(chunk_rows, rows - row_offset);
      raft::update_device(d_row_ids.data_handle(),
                          index_map.data() + row_offset,
                          rows_to_write,
                          stream);

      auto total_values = rows_to_write * cols;
      auto block_size = 256;
      auto grid_size = static_cast<int>((total_values + block_size - 1) / block_size);
      gather_rows_for_cache_kernel<<<grid_size, block_size, 0, stream>>>(
        dataset.data_handle(),
        d_row_ids.data_handle(),
        d_rows.data_handle(),
        rows_to_write,
        cols);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      raft::copy(host_rows.data(), d_rows.data_handle(), total_values, stream);
      raft::resource::sync_stream(res);

      auto payload_bytes = static_cast<std::size_t>(total_values) * sizeof(data_t);
      auto payload_offset = phoenix::kIbinHeaderBytes +
                            static_cast<off_t>(row_offset * cols *
                                               static_cast<int64_t>(sizeof(data_t)));
      if (::pwrite(file, host_rows.data(), payload_bytes, payload_offset) !=
          static_cast<ssize_t>(payload_bytes)) {
        throw std::runtime_error("Cannot write dataset cache payload: " + filename);
      }
    }

    close_file();
    std::cout << "Saving packed dataset cache to " << filename << std::endl;
  } catch (...) {
    close_file();
    throw;
  }
}

template <typename T>
inline void free_preloaded_device_storage(int cuda_device_ordinal, T* storage)
{
  if (storage == nullptr) { return; }
  int original_device = 0;
  auto get_device_ret = cudaGetDevice(&original_device);
  auto restore_device = get_device_ret == cudaSuccess && original_device != cuda_device_ordinal;
  if (restore_device) { cudaSetDevice(cuda_device_ordinal); }
  auto free_ret = cudaFree(storage);
  if (restore_device) { cudaSetDevice(original_device); }
  if (free_ret != cudaSuccess) {
    std::fprintf(stderr,
                 "Warning: cudaFree failed for preloaded VecFlow buffer on CUDA device %d: %s\n",
                 cuda_device_ordinal,
                 cudaGetErrorString(free_ret));
  }
}

template <typename T>
inline auto make_preloaded_device_storage_owner(T* storage, int cuda_device_ordinal)
  -> std::shared_ptr<T>
{
  return std::shared_ptr<T>(storage, [cuda_device_ordinal](T* device_ptr) {
    free_preloaded_device_storage(cuda_device_ordinal, device_ptr);
  });
}

template <typename T>
inline auto make_device_storage_from_host_buffer(shared_resources::configured_raft_resources& res,
                                                 const T* host_storage,
                                                 std::size_t count) -> std::shared_ptr<T>
{
  if (count == 0) { return {}; }
  auto bytes = count * sizeof(T);
  int cuda_device_ordinal = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&cuda_device_ordinal));

  auto stream = raft::resource::get_cuda_stream(res);
  T* device_storage = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&device_storage), bytes));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(device_storage, host_storage, bytes, cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return make_preloaded_device_storage_owner(device_storage, cuda_device_ordinal);
}

template <typename T>
inline auto make_pinned_host_storage_from_host_buffer(const T* host_storage, std::size_t count)
  -> std::shared_ptr<T>
{
  if (count == 0) { return {}; }
  auto bytes = count * sizeof(T);
  T* pinned_storage = nullptr;
  RAFT_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&pinned_storage), bytes));
  std::memcpy(pinned_storage, host_storage, bytes);
  return std::shared_ptr<T>(pinned_storage, [](T* host_ptr) {
    if (host_ptr == nullptr) { return; }
    auto free_ret = cudaFreeHost(host_ptr);
    if (free_ret != cudaSuccess) {
      std::fprintf(stderr,
                   "Warning: cudaFreeHost failed for preloaded VecFlow host buffer: %s\n",
                   cudaGetErrorString(free_ret));
    }
  });
}

template <typename T>
inline auto make_pinned_host_storage_from_device_buffer(shared_resources::configured_raft_resources& res,
                                                        const T* device_storage,
                                                        std::size_t count)
  -> std::shared_ptr<T>
{
  if (count == 0) { return {}; }
  auto bytes = count * sizeof(T);
  auto stream = raft::resource::get_cuda_stream(res);
  T* pinned_storage = nullptr;
  RAFT_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&pinned_storage), bytes));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(pinned_storage, device_storage, bytes, cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return std::shared_ptr<T>(pinned_storage, [](T* host_ptr) {
    if (host_ptr == nullptr) { return; }
    auto free_ret = cudaFreeHost(host_ptr);
    if (free_ret != cudaSuccess) {
      std::fprintf(stderr,
                   "Warning: cudaFreeHost failed for preloaded VecFlow host buffer: %s\n",
                   cudaGetErrorString(free_ret));
    }
  });
}

template <typename data_t>
inline auto gather_dataset_rows_to_device_storage(shared_resources::configured_raft_resources& res,
                                                  raft::device_matrix_view<const data_t, int64_t> dataset,
                                                  const uint32_t* row_ids,
                                                  int64_t row_count)
  -> std::shared_ptr<data_t>
{
  if (row_count <= 0) { return {}; }
  auto dim = dataset.extent(1);
  auto total_values = static_cast<std::size_t>(row_count * dim);
  auto stream = raft::resource::get_cuda_stream(res);
  auto d_row_ids = raft::make_device_vector<uint32_t, int64_t>(res, row_count);
  raft::update_device(d_row_ids.data_handle(), row_ids, row_count, stream);

  int cuda_device_ordinal = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&cuda_device_ordinal));
  data_t* device_storage = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&device_storage), total_values * sizeof(data_t)));

  auto block_size = 256;
  auto grid_size = static_cast<int>((row_count * dim + block_size - 1) / block_size);
  gather_rows_for_cache_kernel<<<grid_size, block_size, 0, stream>>>(dataset.data_handle(),
                                                                      d_row_ids.data_handle(),
                                                                      device_storage,
                                                                      row_count,
                                                                      dim);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return make_preloaded_device_storage_owner(device_storage, cuda_device_ordinal);
}

template <typename data_t>
inline auto gather_dataset_rows_to_pinned_host_storage(
  shared_resources::configured_raft_resources& res,
  raft::device_matrix_view<const data_t, int64_t> dataset,
  const uint32_t* row_ids,
  int64_t row_count) -> std::shared_ptr<data_t>
{
  auto device_storage = gather_dataset_rows_to_device_storage(res, dataset, row_ids, row_count);
  if (device_storage == nullptr) { return {}; }
  return make_pinned_host_storage_from_device_buffer(
    res, device_storage.get(), static_cast<std::size_t>(row_count * dataset.extent(1)));
}

inline void add_graph_to_initial_hbm_cache(const std::shared_ptr<phoenix_label_cache_state>& cache_state,
                                           uint32_t label,
                                           std::size_t bytes,
                                           const std::shared_ptr<uint32_t>& storage)
{
  if (cache_state == nullptr || storage == nullptr || bytes == 0) { return; }
  cache_state->lru_labels.push_front(label);
  cache_state->graphs.emplace(
    label,
    phoenix_label_cache_state::cached_graph_entry{
      storage, bytes, cache_state->lru_labels.begin()});
  cache_state->cached_bytes += bytes;
}

inline void add_graph_to_initial_dram_cache(const std::shared_ptr<phoenix_label_cache_state>& cache_state,
                                            uint32_t label,
                                            std::size_t bytes,
                                            const std::shared_ptr<uint32_t>& storage)
{
  if (cache_state == nullptr || storage == nullptr || bytes == 0) { return; }
  cache_state->host_lru_labels.push_front(label);
  cache_state->host_graphs.emplace(
    label,
    phoenix_label_cache_state::cached_graph_entry{
      storage, bytes, cache_state->host_lru_labels.begin()});
  cache_state->host_cached_bytes += bytes;
}

template <typename data_t>
inline void add_dataset_to_initial_hbm_cache(
  const std::shared_ptr<phoenix_label_dataset_cache_state<data_t>>& cache_state,
  uint32_t label,
  std::size_t bytes,
  const std::shared_ptr<data_t>& storage)
{
  if (cache_state == nullptr || storage == nullptr || bytes == 0) { return; }
  cache_state->lru_labels.push_front(label);
  cache_state->datasets.emplace(
    label,
    typename phoenix_label_dataset_cache_state<data_t>::cached_dataset_entry{
      storage, bytes, cache_state->lru_labels.begin()});
  cache_state->cached_bytes += bytes;
}

template <typename data_t>
inline void add_dataset_to_initial_dram_cache(
  const std::shared_ptr<phoenix_label_dataset_cache_state<data_t>>& cache_state,
  uint32_t label,
  std::size_t bytes,
  const std::shared_ptr<data_t>& storage)
{
  if (cache_state == nullptr || storage == nullptr || bytes == 0) { return; }
  cache_state->host_lru_labels.push_front(label);
  cache_state->host_datasets.emplace(
    label,
    typename phoenix_label_dataset_cache_state<data_t>::cached_dataset_entry{
      storage, bytes, cache_state->host_lru_labels.begin()});
  cache_state->host_cached_bytes += bytes;
}

inline void populate_initial_graph_tiers(shared_resources::configured_raft_resources& res,
                                         const std::shared_ptr<phoenix_label_cache_state>& cache_state,
                                         const std::vector<uint32_t>& host_label_size,
                                         const std::vector<uint32_t>& host_label_offset,
                                         std::size_t hbm_capacity_bytes,
                                         std::size_t dram_capacity_bytes,
                                         const std::string& graph_cache_fname,
                                         int graph_storage_width,
                                         int64_t total_graph_rows,
                                         const uint32_t* host_graph_data)
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  if (cache_state == nullptr || graph_storage_width <= 0 ||
      (hbm_capacity_bytes == 0 && dram_capacity_bytes == 0)) {
    return;
  }

  struct preload_candidate {
    uint32_t label = 0;
    int64_t offset = 0;
    int64_t size = 0;
    std::size_t bytes = 0;
  };

  std::vector<preload_candidate> candidates;
  candidates.reserve(host_label_size.size());
  for (uint32_t label = 0; label < host_label_size.size(); ++label) {
    auto label_size = static_cast<int64_t>(host_label_size[label]);
    if (label_size <= 0) { continue; }
    candidates.push_back(preload_candidate{
      label,
      static_cast<int64_t>(host_label_offset[label]),
      label_size,
      static_cast<std::size_t>(label_size) * static_cast<std::size_t>(graph_storage_width) *
        sizeof(uint32_t)});
  }

  std::sort(candidates.begin(),
            candidates.end(),
            [](preload_candidate const& lhs, preload_candidate const& rhs) {
              if (lhs.bytes != rhs.bytes) { return lhs.bytes > rhs.bytes; }
              return lhs.label < rhs.label;
            });

  auto remaining_hbm = hbm_capacity_bytes;
  auto remaining_dram = dram_capacity_bytes;
  std::size_t hbm_labels = 0;
  std::size_t dram_labels = 0;

  for (auto const& candidate : candidates) {
    auto values = static_cast<std::size_t>(candidate.size) *
                  static_cast<std::size_t>(graph_storage_width);
    if (candidate.bytes <= remaining_hbm) {
      std::shared_ptr<uint32_t> storage;
      if (host_graph_data != nullptr) {
        storage = make_device_storage_from_host_buffer(
          res, host_graph_data + candidate.offset * graph_storage_width, values);
      } else if (!graph_cache_fname.empty()) {
        storage = detail::phoenix::load_ibin_graph_rows_to_device(
          res, graph_cache_fname, total_graph_rows, graph_storage_width, candidate.offset, candidate.size);
      }
      if (storage != nullptr) {
        add_graph_to_initial_hbm_cache(cache_state, candidate.label, candidate.bytes, storage);
        remaining_hbm -= candidate.bytes;
        ++hbm_labels;
      }
      continue;
    }

    if (candidate.bytes <= remaining_dram) {
      std::shared_ptr<uint32_t> storage;
      if (host_graph_data != nullptr) {
        storage = make_pinned_host_storage_from_host_buffer(
          host_graph_data + candidate.offset * graph_storage_width, values);
      } else if (!graph_cache_fname.empty()) {
        auto device_storage = detail::phoenix::load_ibin_graph_rows_to_device(
          res, graph_cache_fname, total_graph_rows, graph_storage_width, candidate.offset, candidate.size);
        storage = make_pinned_host_storage_from_device_buffer(res, device_storage.get(), values);
      }
      if (storage != nullptr) {
        add_graph_to_initial_dram_cache(cache_state, candidate.label, candidate.bytes, storage);
        remaining_dram -= candidate.bytes;
        ++dram_labels;
      }
    }
  }

  if (hbm_labels > 0 || dram_labels > 0) {
    std::cout << "Initial Phoenix graph placement: HBM labels=" << hbm_labels
              << ", DRAM labels=" << dram_labels << ", SSD labels="
              << (candidates.size() - hbm_labels - dram_labels) << std::endl;
  }
#else
  (void)res;
  (void)cache_state;
  (void)host_label_size;
  (void)host_label_offset;
  (void)hbm_capacity_bytes;
  (void)dram_capacity_bytes;
  (void)graph_cache_fname;
  (void)graph_storage_width;
  (void)total_graph_rows;
  (void)host_graph_data;
#endif
}

template <typename data_t>
inline void populate_initial_dataset_tiers(
  shared_resources::configured_raft_resources& res,
  raft::device_matrix_view<const data_t, int64_t> dataset,
  const std::shared_ptr<phoenix_label_dataset_cache_state<data_t>>& cache_state,
  const std::vector<uint32_t>& host_label_size,
  const std::vector<uint32_t>& host_label_offset,
  const std::vector<uint32_t>& host_index_map,
  std::size_t hbm_capacity_bytes,
  std::size_t dram_capacity_bytes)
{
  if (cache_state == nullptr || (hbm_capacity_bytes == 0 && dram_capacity_bytes == 0)) { return; }

  struct preload_candidate {
    uint32_t label = 0;
    int64_t offset = 0;
    int64_t size = 0;
    std::size_t bytes = 0;
  };

  auto dim = dataset.extent(1);
  std::vector<preload_candidate> candidates;
  candidates.reserve(host_label_size.size());
  for (uint32_t label = 0; label < host_label_size.size(); ++label) {
    auto label_size = static_cast<int64_t>(host_label_size[label]);
    if (label_size <= 0) { continue; }
    candidates.push_back(preload_candidate{
      label,
      static_cast<int64_t>(host_label_offset[label]),
      label_size,
      static_cast<std::size_t>(label_size) * static_cast<std::size_t>(dim) * sizeof(data_t)});
  }

  std::sort(candidates.begin(),
            candidates.end(),
            [](preload_candidate const& lhs, preload_candidate const& rhs) {
              if (lhs.bytes != rhs.bytes) { return lhs.bytes > rhs.bytes; }
              return lhs.label < rhs.label;
            });

  auto remaining_hbm = hbm_capacity_bytes;
  auto remaining_dram = dram_capacity_bytes;
  std::size_t hbm_labels = 0;
  std::size_t dram_labels = 0;

  for (auto const& candidate : candidates) {
    auto row_ids = host_index_map.data() + candidate.offset;
    if (candidate.bytes <= remaining_hbm) {
      auto storage =
        gather_dataset_rows_to_device_storage(res, dataset, row_ids, candidate.size);
      if (storage != nullptr) {
        add_dataset_to_initial_hbm_cache(cache_state, candidate.label, candidate.bytes, storage);
        remaining_hbm -= candidate.bytes;
        ++hbm_labels;
      }
      continue;
    }

    if (candidate.bytes <= remaining_dram) {
      auto storage =
        gather_dataset_rows_to_pinned_host_storage(res, dataset, row_ids, candidate.size);
      if (storage != nullptr) {
        add_dataset_to_initial_dram_cache(cache_state, candidate.label, candidate.bytes, storage);
        remaining_dram -= candidate.bytes;
        ++dram_labels;
      }
    }
  }

  if (hbm_labels > 0 || dram_labels > 0) {
    std::cout << "Initial Phoenix dataset placement: HBM labels=" << hbm_labels
              << ", DRAM labels=" << dram_labels << ", SSD labels="
              << (candidates.size() - hbm_labels - dram_labels) << std::endl;
  }
}

template <typename data_t>
auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const data_t, int64_t> dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname,
           const std::string& bfs_fname,
           bool force_rebuild,
           graph_builder_type graph_builder,
           int tagore_iterations) -> cuvs::neighbors::vecflow::index<data_t> {

  std::vector<int> cat_freq(data_label_vecs.size(), 0);

  int max_label = 0;
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    for (size_t j = 0; j < data_label_vecs[i].size(); j++) {
      max_label = std::max(max_label, data_label_vecs[i][j]);
    }
  }

  cat_freq.resize(max_label + 1, 0);
  std::vector<std::vector<int>> label_data_vecs(max_label + 1);
  for (size_t i = 0; i < data_label_vecs.size(); i++) {
    for (size_t j = 0; j < data_label_vecs[i].size(); j++) {
      cat_freq[data_label_vecs[i][j]] += 1;
      label_data_vecs[data_label_vecs[i][j]].push_back(i);
    }
  }

  // Prepare metadata
  uint32_t label_number = max_label + 1;
  int64_t cagra_total_rows = 0;
  int64_t bfs_total_rows = 0;
  int cagra_labels = 0;
  int bfs_labels = 0;
  std::vector<uint32_t> host_cagra_label_size(label_number);
  std::vector<uint32_t> host_cagra_label_offset(label_number);
  std::vector<uint32_t> host_bfs_label_size(label_number);
  std::vector<uint32_t> host_bfs_label_offset(label_number);
  std::vector<uint32_t> host_cat_freq(label_number);
  for (uint32_t i = 0; i < label_number; i++) {
    host_cat_freq[i] = cat_freq[i];
    auto n_rows = label_data_vecs[i].size();
    if (n_rows == 0) {
      host_cagra_label_size[i] = 0;
      host_cagra_label_offset[i] = cagra_total_rows;
      host_bfs_label_size[i] = 0;
      host_bfs_label_offset[i] = bfs_total_rows;
      continue;
    }
    if (cat_freq[i] > specificity_threshold) {
      host_cagra_label_size[i] = n_rows;
      host_cagra_label_offset[i] = cagra_total_rows;
      host_bfs_label_size[i] = 0;
      host_bfs_label_offset[i] = bfs_total_rows;
      cagra_total_rows += n_rows;
      cagra_labels++;
    } else {
      host_cagra_label_size[i] = 0;
      host_cagra_label_offset[i] = cagra_total_rows;
      host_bfs_label_size[i] = n_rows;
      host_bfs_label_offset[i] = bfs_total_rows;
      bfs_total_rows += n_rows;
      bfs_labels++;
    }
  }

  std::vector<uint32_t> host_cagra_index_map(cagra_total_rows);
  std::vector<uint32_t> host_bfs_index_map(bfs_total_rows);
  uint32_t bfs_iter = 0;
  uint32_t cagra_iter = 0;
  for (uint32_t i = 0; i < label_number; i ++) {
    if (cat_freq[i] > specificity_threshold) {
      for (uint32_t j = 0; j < label_data_vecs[i].size(); j ++) {
        host_cagra_index_map[cagra_iter] = label_data_vecs[i][j];
        cagra_iter ++;
      }
    }
    else {
      for (uint32_t j = 0; j < label_data_vecs[i].size(); j ++) {
        host_bfs_index_map[bfs_iter] = label_data_vecs[i][j];
        bfs_iter ++;
      }
    }
  }

  auto cagra_index_map = raft::make_device_vector<uint32_t, int64_t>(res, cagra_total_rows);
  auto cagra_label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto cagra_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto bfs_label_size = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
  auto d_cat_freq = raft::make_device_vector<uint32_t, int64_t>(res, label_number);

  raft::update_device(cagra_label_size.data_handle(), 
                      host_cagra_label_size.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(cagra_label_offset.data_handle(),
                      host_cagra_label_offset.data(),
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(cagra_index_map.data_handle(),
                      host_cagra_index_map.data(),
                      cagra_total_rows,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(bfs_label_size.data_handle(), 
                      host_bfs_label_size.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));
  raft::update_device(d_cat_freq.data_handle(), 
                      host_cat_freq.data(), 
                      label_number,
                      raft::resource::get_cuda_stream(res));

  auto ivf_graph_index = cagra::index<data_t, uint32_t>(res);
  auto ivf_bfs_index = ivf_flat::index<data_t, int64_t>(res,
                                                        cuvs::distance::DistanceType::L2Unexpanded,
                                                        label_number,
                                                        false,
                                                        true,
                                                        dataset.extent(1));
  std::shared_ptr<uint32_t> cagra_graph_storage;
  auto dataset_cache_fname =
    !graph_fname.empty() ? append_suffix_to_filename(graph_fname, "_dataset") : std::string{};
  bool deferred_phoenix_label_load = false;
  bool deferred_phoenix_dataset_load = false;
  auto use_tagore_builder = graph_builder != graph_builder_type::CAGRA;
  auto enable_phoenix_label_load = detail::phoenix::use_phoenix_label_load();
  auto graph_storage_width = graph_degree;
  auto phoenix_cache_state = std::make_shared<phoenix_label_cache_state>();
  phoenix_cache_state->access_counts.resize(label_number, 0);
  auto phoenix_dataset_cache_state =
    std::make_shared<phoenix_label_dataset_cache_state<data_t>>();

  // Index Information  
  std::cout << "\n=== Index Information ===" << std::endl;

  if (cagra_labels > 0) {
    if (enable_phoenix_label_load && graph_fname.empty()) {
      throw std::runtime_error(
        "Phoenix label-load path requires a non-empty VecFlow graph cache filename.");
    }
    deferred_phoenix_dataset_load = enable_phoenix_label_load;
    if (!deferred_phoenix_dataset_load) {
      ivf_graph_index.update_dataset(res, raft::make_const_mdspan(dataset));
    }
    auto host_final_graph =
      raft::make_host_matrix<uint32_t, int64_t>(cagra_total_rows, graph_storage_width);
    auto use_cached_graph = !graph_fname.empty() && std::filesystem::exists(graph_fname) && !force_rebuild;
    auto use_cached_dataset =
      !dataset_cache_fname.empty() && std::filesystem::exists(dataset_cache_fname) && !force_rebuild;
    if (use_cached_graph) {
      auto [cached_rows, cached_cols] = read_ibin_shape(graph_fname);
      if (cached_rows != cagra_total_rows || cached_cols != graph_storage_width) {
        std::cout << "Cached IVF-Graph at " << graph_fname << " has shape [" << cached_rows
                  << " x " << cached_cols << "], expected [" << cagra_total_rows << " x "
                  << graph_storage_width << "]. Rebuilding." << std::endl;
        use_cached_graph = false;
      }
    }
    if (use_cached_dataset) {
      auto [cached_rows, cached_cols] = read_ibin_shape(dataset_cache_fname);
      if (cached_rows != cagra_total_rows || cached_cols != dataset.extent(1)) {
        std::cout << "Cached packed dataset at " << dataset_cache_fname << " has shape ["
                  << cached_rows << " x " << cached_cols << "], expected ["
                  << cagra_total_rows << " x " << dataset.extent(1) << "]. Rebuilding."
                  << std::endl;
        use_cached_dataset = false;
      }
    }
    if (use_cached_graph) {
      if (enable_phoenix_label_load) {
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
        deferred_phoenix_label_load = true;
        std::cout << "Deferring IVF-Graph loading to Phoenix label-load path from "
                  << graph_fname << std::endl;
#else
        throw std::runtime_error(
          "CUVS_VECFLOW_USE_PHOENIX_LABEL_LOAD is set, but VecFlow was built without Phoenix support.");
#endif
      } else if (detail::phoenix::use_phoenix_graph_load()) {
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
        cagra_graph_storage = detail::phoenix::load_ibin_graph_to_device(
          res, graph_fname, cagra_total_rows, graph_storage_width);
        auto device_graph_view = raft::make_device_matrix_view<const uint32_t, int64_t, raft::row_major>(
          cagra_graph_storage.get(), cagra_total_rows, graph_storage_width);
        ivf_graph_index.update_graph(res, device_graph_view);
#else
        throw std::runtime_error(
          "CUVS_VECFLOW_USE_PHOENIX_GRAPH_LOAD is set, but VecFlow was built without Phoenix support.");
#endif
      } else {
        load_matrix_from_ibin(graph_fname, host_final_graph.view());
        ivf_graph_index.update_graph(res, raft::make_const_mdspan(host_final_graph.view()));
      }
    } else {
      std::cout << "Building IVF-Graph index from scratch ..." << std::endl;
      auto cagra_start_time = std::chrono::high_resolution_clock::now();
      int build_device_id = 0;
      RAFT_CUDA_TRY(cudaGetDevice(&build_device_id));
      int optimal_threads = 32;
      omp_set_num_threads(optimal_threads);
      std::atomic<size_t> completed_work{0};
      #pragma omp parallel for num_threads(optimal_threads)
      for (uint32_t i = 0; i < label_number; i++) {
        if (host_cagra_label_size[i] == 0 || label_data_vecs[i].size() == 0) continue;

        int thread_id = omp_get_thread_num();
        shared_resources::thread_id = thread_id;
        shared_resources::n_threads = optimal_threads;
        RAFT_CUDA_TRY(cudaSetDevice(build_device_id));
        auto thread_resources = res;
        cudaStream_t thread_stream = thread_resources.get_sync_stream();

        // Progress calculation and display
        #pragma omp critical
        {
          completed_work += label_data_vecs[i].size();;
          float progress = (float)completed_work / cagra_total_rows * 100;
          std::cout << "\rProgress: [";
          int pos = progress / 2;
          for (int j = 0; j < 50; j++) {
              if (j < pos) std::cout << "=";
              else if (j == pos) std::cout << ">";
              else std::cout << " ";
          }
          std::cout << "] " << std::fixed << std::setprecision(1) << progress << "% "
                    << "Label: " << i << " Size: " << label_data_vecs[i].size() << std::flush;
        }

        auto filtered_dataset = raft::make_device_matrix<data_t, int64_t>(
          thread_resources, label_data_vecs[i].size(), dataset.extent(1));
        raft::resource::sync_stream(thread_resources);
        for (uint64_t j = 0; j < label_data_vecs[i].size(); j++) {
          raft::copy_async(filtered_dataset.data_handle() + j * dataset.extent(1), 
                           dataset.data_handle() + static_cast<uint64_t>(label_data_vecs[i][j]) * dataset.extent(1), 
                           dataset.extent(1), 
                           thread_stream);
        }
        raft::resource::sync_stream(thread_resources);
        
        if (use_tagore_builder) {
#ifdef CUVS_VECFLOW_TAGORE_ENABLED
          if constexpr (!std::is_same_v<data_t, float>) {
            throw std::runtime_error("Tagore VecFlow builder is currently only implemented for float data.");
          } else {
            detail::tagore::validate_build_inputs(filtered_dataset.extent(0),
                                                  filtered_dataset.extent(1),
                                                  static_cast<unsigned>(graph_degree),
                                                  std::to_string(i));
            auto graph_view = raft::make_host_matrix_view<uint32_t, int64_t>(
              host_final_graph.data_handle() +
                static_cast<int64_t>(host_cagra_label_offset[i]) * graph_storage_width,
              host_cagra_label_size[i],
              graph_storage_width);
            detail::tagore::build_cagra_compatible_graph(thread_resources,
                                                         raft::make_const_mdspan(
                                                           filtered_dataset.view()),
                                                         static_cast<unsigned>(graph_degree),
                                                         static_cast<unsigned>(tagore_iterations),
                                                         graph_view);
          }
#else
          throw std::runtime_error("VecFlow was built without Tagore support.");
#endif
        } else {
          cagra::index_params index_params;
          index_params.intermediate_graph_degree = graph_degree * 2;
          index_params.graph_degree = graph_degree;
          index_params.attach_dataset_on_build = false;
          auto index =
            cagra::build(thread_resources, index_params, raft::make_const_mdspan(filtered_dataset.view()));

          raft::copy(host_final_graph.data_handle() +
                       static_cast<int64_t>(host_cagra_label_offset[i]) * graph_storage_width,
                     index.graph().data_handle(),
                     host_cagra_label_size[i] * graph_degree,
                     thread_stream);
          raft::resource::sync_stream(thread_resources);
        }
      }
      auto cagra_end_time = std::chrono::high_resolution_clock::now();
      auto cagra_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cagra_end_time - cagra_start_time);
      std::cout << "\nIVF-Graph index building time: " << cagra_duration.count() << " ms" << std::endl;
      if (enable_phoenix_label_load) {
        deferred_phoenix_label_load = true;
        std::cout << "Deferring newly built IVF-Graph attachment to Phoenix label-load path"
                  << std::endl;
      } else {
        ivf_graph_index.update_graph(res, raft::make_const_mdspan(host_final_graph.view()));
      }
      if (!graph_fname.empty()) {
        save_matrix_to_ibin(graph_fname, host_final_graph.view());
      }
    }

    if (!dataset_cache_fname.empty() && !use_cached_dataset) {
      write_packed_dataset_cache(res, dataset, host_cagra_index_map, dataset_cache_fname);
    }

    if (enable_phoenix_label_load) {
      populate_initial_graph_tiers(res,
                                   phoenix_cache_state,
                                   host_cagra_label_size,
                                   host_cagra_label_offset,
                                   detail::phoenix::phoenix_label_cache_bytes(),
                                   detail::phoenix::phoenix_label_dram_cache_bytes(),
                                   graph_fname,
                                   graph_storage_width,
                                   cagra_total_rows,
                                   use_cached_graph ? nullptr : host_final_graph.data_handle());
      populate_initial_dataset_tiers(res,
                                     dataset,
                                     phoenix_dataset_cache_state,
                                     host_cagra_label_size,
                                     host_cagra_label_offset,
                                     host_cagra_index_map,
                                     detail::phoenix::phoenix_label_dataset_cache_bytes(),
                                     detail::phoenix::phoenix_label_dataset_dram_cache_bytes());

      bool graph_fully_resident_in_hbm = false;
      {
        std::lock_guard<std::mutex> lock(phoenix_cache_state->mutex);
        graph_fully_resident_in_hbm =
          static_cast<int>(phoenix_cache_state->graphs.size()) == cagra_labels &&
          phoenix_cache_state->host_graphs.empty();
      }

      if (graph_fully_resident_in_hbm && cagra_total_rows > 0) {
        if (use_cached_graph) { load_matrix_from_ibin(graph_fname, host_final_graph.view()); }
        ivf_graph_index.update_graph(res, raft::make_const_mdspan(host_final_graph.view()));
        ivf_graph_index.update_dataset(res, raft::make_const_mdspan(dataset));
        deferred_phoenix_label_load   = false;
        deferred_phoenix_dataset_load = false;

        {
          std::lock_guard<std::mutex> lock(phoenix_cache_state->mutex);
          phoenix_cache_state->graphs.clear();
          phoenix_cache_state->lru_labels.clear();
          phoenix_cache_state->cached_bytes = 0;
          phoenix_cache_state->host_graphs.clear();
          phoenix_cache_state->host_lru_labels.clear();
          phoenix_cache_state->host_cached_bytes = 0;
        }
        {
          std::lock_guard<std::mutex> lock(phoenix_dataset_cache_state->mutex);
          phoenix_dataset_cache_state->datasets.clear();
          phoenix_dataset_cache_state->lru_labels.clear();
          phoenix_dataset_cache_state->cached_bytes = 0;
          phoenix_dataset_cache_state->host_datasets.clear();
          phoenix_dataset_cache_state->host_lru_labels.clear();
          phoenix_dataset_cache_state->host_cached_bytes = 0;
        }

        std::cout << "Phoenix label-load graph fully fits in HBM; attaching full IVF-Graph for "
                     "direct search."
                  << std::endl;
      }
    }
  }

  if (bfs_labels > 0) {
    if (std::filesystem::exists(bfs_fname) && !force_rebuild) {
      std::cout << "Loading IVF-BFS index from " << bfs_fname << std::endl;
      ivf_flat::deserialize(res, bfs_fname, &ivf_bfs_index);
    } else {
      std::cout << "Building IVF-BFS index from scratch ..." << std::endl;
      auto bfs_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, label_number);
      auto bfs_index_map = raft::make_device_vector<uint32_t, int64_t>(res, bfs_total_rows);
      raft::update_device(bfs_label_offset.data_handle(),
                          host_bfs_label_offset.data(),
                          label_number,
                          raft::resource::get_cuda_stream(res));
      raft::update_device(bfs_index_map.data_handle(),
                          host_bfs_index_map.data(),
                          bfs_total_rows,
                          raft::resource::get_cuda_stream(res));
      auto bfs_start_time = std::chrono::high_resolution_clock::now();                 
      build_filtered_bfs(res,
                         &ivf_bfs_index,
                         dataset,
                         bfs_index_map.view(),
                         bfs_label_size.view(),
                         bfs_label_offset.view());
      auto bfs_end_time = std::chrono::high_resolution_clock::now();
      auto bfs_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bfs_end_time - bfs_start_time);
      std::cout << "IVF-BFS graph building time: " << bfs_duration.count() << " ms" << std::endl;
      if (!bfs_fname.empty()) {
        ivf_flat::serialize(res, bfs_fname, ivf_bfs_index);
        std::cout << "Saving IVF-BFS index to " << bfs_fname << std::endl;
      }
    }
  }
  
  // IVF-Graph statistics
  std::cout << "\nIVF-Graph Index Stats:" << std::endl;
  auto total_vectors_to_report =
    deferred_phoenix_label_load ? cagra_total_rows : static_cast<int64_t>(ivf_graph_index.size());
  std::cout << "  Total vectors:  " << total_vectors_to_report << std::endl;
  std::cout << "  Number of labels: " << cagra_labels << std::endl;
  auto graph_rows_to_report =
    deferred_phoenix_label_load ? cagra_total_rows : ivf_graph_index.graph().extent(0);
  auto graph_width_to_report =
    deferred_phoenix_label_load ? graph_storage_width : ivf_graph_index.graph_degree();
  std::cout << "  Graph size:     [" << graph_rows_to_report << " × "
            << graph_width_to_report << "]" << std::endl;
  std::cout << "  Graph degree:   " << graph_degree << std::endl;
  if (static_cast<int>(graph_width_to_report) != graph_degree) {
    std::cout << "  Storage width:  " << graph_width_to_report << std::endl;
  }
  if (deferred_phoenix_dataset_load && !dataset_cache_fname.empty()) {
    std::cout << "  Dataset cache:  " << dataset_cache_fname << std::endl;
  }
  // IVF statistics
  std::cout << "\nIVF-BFS Index Stats:" << std::endl;
  std::cout << "  Number of labels: " << bfs_labels << std::endl;
  std::cout << "  Number of rows:  " << bfs_total_rows << std::endl;

  return cuvs::neighbors::vecflow::index<data_t>{
    std::move(ivf_graph_index),
    std::move(ivf_bfs_index),
    specificity_threshold,
    std::move(cagra_index_map),
    std::move(cagra_label_size),
    std::move(cagra_label_offset),
    std::move(bfs_label_size),
    std::move(d_cat_freq),
    std::move(cagra_graph_storage),
    std::move(host_cagra_label_size),
    std::move(host_cagra_label_offset),
    graph_fname,
    graph_degree,
    graph_storage_width,
    dataset_cache_fname,
    static_cast<int>(dataset.extent(1)),
    graph_builder,
    std::move(phoenix_cache_state),
    std::move(phoenix_dataset_cache_state),
    detail::phoenix::phoenix_label_cache_bytes(),
    detail::phoenix::phoenix_label_dram_cache_bytes(),
    detail::phoenix::phoenix_label_dataset_cache_bytes(),
    detail::phoenix::phoenix_label_dataset_dram_cache_bytes(),
    detail::phoenix::phoenix_label_rebalance_interval_queries()};
}

}  // namespace detail

template<typename data_t>
auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const data_t, int64_t> dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname,
           const std::string& bfs_fname,
           bool force_rebuild,
           graph_builder_type graph_builder,
           int tagore_iterations) -> cuvs::neighbors::vecflow::index<data_t>
{
  return cuvs::neighbors::vecflow::detail::build<data_t>(
    res,
    dataset,
    data_label_vecs,
    graph_degree,
    specificity_threshold,
    graph_fname,
    bfs_fname,
    force_rebuild,
    graph_builder,
    tagore_iterations);
} 

template <typename data_t>
auto build_multi_gpu_impl(shared_resources::configured_raft_resources& res,
                          raft::device_matrix_view<const data_t, int64_t> dataset,
                          const std::vector<std::vector<int>>& data_label_vecs,
                          int graph_degree,
                          int specificity_threshold,
                          const multi_gpu_params& mg_params,
                          const std::string& graph_fname,
                          const std::string& bfs_fname,
                          bool force_rebuild,
                          graph_builder_type graph_builder,
                          int tagore_iterations)
  -> cuvs::neighbors::vecflow::multi_gpu_index<data_t>
{
  auto host_dataset = detail::multi_gpu::copy_device_matrix_to_host(res, dataset);
  auto label_counts = detail::multi_gpu::compute_label_counts(data_label_vecs);
  auto assignment   = detail::multi_gpu::assign_labels_to_workers(label_counts, mg_params);
  auto total_values = dataset.extent(0) * dataset.extent(1);

  int original_device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&original_device));

  cuvs::neighbors::vecflow::multi_gpu_index<data_t> mg_index;
  mg_index.workers.resize(mg_params.device_ids.size());
  mg_index.label_to_worker        = assignment.label_to_worker;
  mg_index.worker_loads           = assignment.worker_loads;
  mg_index.graph_degree           = graph_degree;
  mg_index.specificity_threshold  = specificity_threshold;
  mg_index.graph_builder          = graph_builder;

  try {
    std::vector<std::string> worker_errors;
    std::mutex worker_error_mutex;
    std::vector<std::thread> build_workers;
    build_workers.reserve(mg_params.device_ids.size());

    for (std::size_t worker_idx = 0; worker_idx < mg_params.device_ids.size(); ++worker_idx) {
      auto& worker      = mg_index.workers[worker_idx];
      worker.device_id  = mg_params.device_ids[worker_idx];
      worker.labels     = assignment.worker_labels[worker_idx];
      if (worker.labels.empty()) { continue; }
      build_workers.emplace_back([&, worker_idx]() {
        try {
          auto device_id = mg_params.device_ids[worker_idx];
          RAFT_CUDA_TRY(cudaSetDevice(device_id));

          auto& thread_worker = mg_index.workers[worker_idx];
          shared_resources::thread_id = static_cast<int>(worker_idx);
          shared_resources::n_threads = static_cast<int>(mg_params.device_ids.size());
          thread_worker.resources = std::make_shared<shared_resources::configured_raft_resources>();
          auto& worker_res = *thread_worker.resources;

          thread_worker.dataset.emplace(
            raft::make_device_matrix<data_t, int64_t>(worker_res, dataset.extent(0), dataset.extent(1)));
          raft::copy(thread_worker.dataset->data_handle(),
                     host_dataset.data(),
                     total_values,
                     raft::resource::get_cuda_stream(worker_res));
          raft::resource::sync_stream(worker_res);

          auto filtered_labels =
            detail::multi_gpu::filter_labels_by_owner(data_label_vecs, thread_worker.labels);
          auto worker_graph_fname =
            detail::multi_gpu::make_worker_cache_filename(graph_fname, device_id);
          auto worker_bfs_fname =
            detail::multi_gpu::make_worker_cache_filename(bfs_fname, device_id);

          thread_worker.index.emplace(cuvs::neighbors::vecflow::build(
            worker_res,
            raft::make_const_mdspan(thread_worker.dataset->view()),
            filtered_labels,
            graph_degree,
            specificity_threshold,
            worker_graph_fname,
            worker_bfs_fname,
            force_rebuild,
            graph_builder,
            tagore_iterations));
          raft::resource::sync_stream(worker_res);
        } catch (std::exception const& e) {
          std::lock_guard<std::mutex> lock(worker_error_mutex);
          worker_errors.push_back("worker " + std::to_string(worker_idx) + ": " + e.what());
        } catch (...) {
          std::lock_guard<std::mutex> lock(worker_error_mutex);
          worker_errors.push_back("worker " + std::to_string(worker_idx) +
                                  ": unknown non-std exception");
        }
      });
    }

    for (auto& build_worker : build_workers) { build_worker.join(); }
    if (!worker_errors.empty()) {
      std::string message = "build_multi_gpu worker failures:";
      for (auto const& error : worker_errors) {
        message += "\n";
        message += error;
      }
      throw std::runtime_error(message);
    }
    RAFT_CUDA_TRY(cudaSetDevice(original_device));
  } catch (...) {
    cudaSetDevice(original_device);
    throw;
  }

  return mg_index;
}

} // namespace cuvs::neighbors::vecflow
