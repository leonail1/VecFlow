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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <omp.h>
#include <array>
#include <chrono>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <future>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "multi_gpu.cuh"
#include "vecflow_common.cuh"
#include "phoenix_graph_load.cuh"


namespace cuvs::neighbors::vecflow {

namespace detail {

inline bool vecflow_verbose_logging()
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return detail::phoenix::verbose_logging();
#else
  return false;
#endif
}

template <typename... Args>
inline void vecflow_log(Args&&... args)
{
  if (!vecflow_verbose_logging()) { return; }
  ((std::cout << std::forward<Args>(args)), ...);
  std::cout << std::endl;
}

template <typename data_t>
__global__ void gather_rows_kernel(const data_t* source,
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

template <typename idx_t>
__global__ void scatter_group_results_kernel(idx_t* neighbors,
                                             float* distances,
                                             const idx_t* neighbor_src,
                                             const float* distance_src,
                                             const uint32_t* indices,
                                             int n_queries,
                                             int topk)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto n_elements = n_queries * topk;
  if (tid >= n_elements) { return; }

  auto query_idx = tid / topk;
  auto k_idx = tid % topk;
  auto src_offset = query_idx * topk + k_idx;
  auto dst_offset = indices[query_idx] * topk + k_idx;

  neighbors[dst_offset] = neighbor_src[src_offset];
  distances[dst_offset] = distance_src[src_offset];
}

__device__ inline bool device_binary_search_u32(const uint32_t* data,
                                                uint32_t size,
                                                uint32_t target)
{
  int64_t left = 0;
  int64_t right = static_cast<int64_t>(size);
  while (left < right) {
    auto mid = left + ((right - left) >> 1);
    auto value = data[mid];
    if (value < target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left < static_cast<int64_t>(size) && data[left] == target;
}

__global__ void flatten_subquery_results_kernel(const uint32_t* subquery_to_query,
                                                const uint32_t* sub_neighbors,
                                                const float* sub_distances,
                                                uint32_t* flat_query_ids,
                                                uint32_t* flat_sample_ids,
                                                float* flat_distances,
                                                uint32_t query_id_base,
                                                int64_t subquery_count,
                                                int topk)
{
  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = subquery_count * static_cast<int64_t>(topk);
  if (tid >= total) { return; }
  auto subquery = tid / topk;
  flat_query_ids[tid] = subquery_to_query[subquery] - query_id_base;
  flat_sample_ids[tid] = sub_neighbors[tid];
  flat_distances[tid] = sub_distances[tid];
}

__global__ void flatten_output_results_kernel(const uint32_t* neighbors,
                                              const float* distances,
                                              uint32_t* flat_query_ids,
                                              uint32_t* flat_sample_ids,
                                              float* flat_distances,
                                              int64_t query_count,
                                              int topk)
{
  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = query_count * static_cast<int64_t>(topk);
  if (tid >= total) { return; }
  auto query = tid / topk;
  flat_query_ids[tid] = static_cast<uint32_t>(query);
  flat_sample_ids[tid] = neighbors[tid];
  flat_distances[tid] = distances[tid];
}

__global__ void pack_query_sample_keys_kernel(const uint32_t* query_ids,
                                              const uint32_t* sample_ids,
                                              uint64_t* keys,
                                              int64_t count)
{
  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= count) { return; }
  keys[tid] = (static_cast<uint64_t>(query_ids[tid]) << 32) |
              static_cast<uint64_t>(sample_ids[tid]);
}

__global__ void unpack_query_sample_keys_kernel(const uint64_t* keys,
                                                uint32_t* query_ids,
                                                uint32_t* sample_ids,
                                                int64_t count)
{
  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= count) { return; }
  query_ids[tid] = static_cast<uint32_t>(keys[tid] >> 32);
  sample_ids[tid] = static_cast<uint32_t>(keys[tid] & 0xffffffffULL);
}

__global__ void scatter_sorted_topk_kernel(const int64_t* query_offsets,
                                           const uint32_t* sorted_samples,
                                           const float* sorted_distances,
                                           int64_t query_count,
                                           int topk,
                                           uint32_t* neighbors,
                                           float* distances)
{
  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = query_count * static_cast<int64_t>(topk);
  if (tid >= total) { return; }
  auto query = tid / topk;
  auto rank = tid % topk;
  auto begin = query_offsets[query];
  auto end = query_offsets[query + 1];
  auto idx = begin + rank;
  if (idx >= end) { return; }
  neighbors[tid] = sorted_samples[idx];
  distances[tid] = sorted_distances[idx];
}

__global__ void count_valid_results_kernel(const uint32_t* neighbors,
                                           int64_t query_count,
                                           int topk,
                                           uint32_t* counts)
{
  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = query_count * static_cast<int64_t>(topk);
  if (tid >= total) { return; }
  if (neighbors[tid] == UINT32_MAX) { return; }
  auto query = tid / topk;
  atomicAdd(counts + query, 1u);
}

__global__ void filter_results_by_membership_kernel(
  const int64_t* query_label_offsets,
  const uint32_t* query_label_indices,
  const uint32_t* subquery_to_query,
  const uint32_t* cagra_label_size,
  const uint32_t* cagra_label_offset,
  const uint32_t* cagra_index_map,
  const uint32_t* bfs_label_size,
  const uint32_t* bfs_label_offset,
  const uint32_t* bfs_index_map,
  uint32_t* neighbors,
  float* distances,
  int64_t subquery_count,
  int topk,
  uint32_t* valid_counts)
{
  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = subquery_count * static_cast<int64_t>(topk);
  if (tid >= total) { return; }

  auto sample_id = neighbors[tid];
  if (sample_id == UINT32_MAX) { return; }

  auto subquery = tid / topk;
  auto query_id = subquery_to_query[subquery];
  auto label_begin = query_label_offsets[query_id];
  auto label_end = query_label_offsets[query_id + 1];
  auto valid = true;
  for (auto label_idx = label_begin; label_idx < label_end; ++label_idx) {
    auto label = query_label_indices[label_idx];
    const uint32_t* members = nullptr;
    uint32_t member_count = 0;
    if (bfs_label_size != nullptr && bfs_label_offset != nullptr && bfs_index_map != nullptr &&
        bfs_label_size[label] > 0) {
      member_count = bfs_label_size[label];
      members = bfs_index_map + bfs_label_offset[label];
    } else if (cagra_label_size != nullptr && cagra_label_offset != nullptr &&
               cagra_index_map != nullptr && cagra_label_size[label] > 0) {
      member_count = cagra_label_size[label];
      members = cagra_index_map + cagra_label_offset[label];
    } else {
      valid = false;
      break;
    }
    if (!device_binary_search_u32(members, member_count, sample_id)) {
      valid = false;
      break;
    }
  }

  if (!valid) {
    neighbors[tid] = UINT32_MAX;
    distances[tid] = std::numeric_limits<float>::infinity();
    return;
  }
  atomicAdd(valid_counts + query_id, 1u);
}

template <typename T>
inline void free_device_storage(int cuda_device_ordinal, T* device_ptr)
{
  if (device_ptr == nullptr) { return; }
  int original_device = 0;
  auto get_device_ret = cudaGetDevice(&original_device);
  auto restore_device = get_device_ret == cudaSuccess && original_device != cuda_device_ordinal;
  if (restore_device) { cudaSetDevice(cuda_device_ordinal); }
  auto free_ret = cudaFree(device_ptr);
  if (restore_device) { cudaSetDevice(original_device); }
  if (free_ret != cudaSuccess) {
    std::fprintf(stderr,
                 "Warning: cudaFree failed for VecFlow graph buffer on CUDA device %d: %s\n",
                 cuda_device_ordinal,
                 cudaGetErrorString(free_ret));
  }
}

template <typename T>
inline auto make_device_storage_owner(T* device_storage, int cuda_device_ordinal)
  -> std::shared_ptr<T>
{
  return std::shared_ptr<T>(device_storage, [cuda_device_ordinal](T* device_ptr) {
    free_device_storage(cuda_device_ordinal, device_ptr);
  });
}

template <typename T>
class reusable_device_buffer {
 public:
  reusable_device_buffer() = default;

  ~reusable_device_buffer()
  {
    if (storage_ != nullptr) {
      free_device_storage(cuda_device_ordinal_, storage_);
      storage_        = nullptr;
      capacity_bytes_ = 0;
    }
  }

  reusable_device_buffer(const reusable_device_buffer&) = delete;
  auto operator=(const reusable_device_buffer&) -> reusable_device_buffer& = delete;
  reusable_device_buffer(reusable_device_buffer&&) = delete;
  auto operator=(reusable_device_buffer&&) -> reusable_device_buffer& = delete;

  auto copy_from_host(shared_resources::configured_raft_resources& res,
                      const T* host_storage,
                      std::size_t bytes) -> std::shared_ptr<T>
  {
    auto stream = raft::resource::get_cuda_stream(res);
    ensure_capacity(bytes);
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(storage_, host_storage, bytes, cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    return std::shared_ptr<T>(storage_, [](T*) {});
  }

 private:
  void ensure_capacity(std::size_t requested_bytes)
  {
    if (capacity_bytes_ >= requested_bytes && storage_ != nullptr) { return; }
    if (storage_ != nullptr) {
      free_device_storage(cuda_device_ordinal_, storage_);
      storage_        = nullptr;
      capacity_bytes_ = 0;
    }
    RAFT_CUDA_TRY(cudaGetDevice(&cuda_device_ordinal_));
    RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&storage_), requested_bytes));
    capacity_bytes_ = requested_bytes;
  }

  int cuda_device_ordinal_ = 0;
  T* storage_              = nullptr;
  std::size_t capacity_bytes_ = 0;
};

template <typename T>
inline auto make_device_storage_from_host(shared_resources::configured_raft_resources& res,
                                          const T* host_storage,
                                          std::size_t bytes)
  -> std::shared_ptr<T>
{
  auto stream = raft::resource::get_cuda_stream(res);
  int cuda_device_ordinal = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&cuda_device_ordinal));

  T* device_storage = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&device_storage), bytes));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(device_storage, host_storage, bytes, cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return make_device_storage_owner(device_storage, cuda_device_ordinal);
}

template <typename T>
inline auto make_pinned_host_storage_from_device(shared_resources::configured_raft_resources& res,
                                                 const T* device_storage,
                                                 std::size_t bytes)
  -> std::shared_ptr<T>
{
  auto stream = raft::resource::get_cuda_stream(res);
  T* host_storage = nullptr;
  RAFT_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&host_storage), bytes));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(host_storage, device_storage, bytes, cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return std::shared_ptr<T>(host_storage, [](T* host_ptr) {
    if (host_ptr == nullptr) { return; }
    auto free_ret = cudaFreeHost(host_ptr);
    if (free_ret != cudaSuccess) {
      std::fprintf(stderr, "Warning: cudaFreeHost failed for VecFlow graph buffer: %s\n",
                   cudaGetErrorString(free_ret));
    }
  });
}

template <typename T>
inline auto load_ibin_rows_to_pinned_host(const std::string& filename,
                                          int64_t expected_rows,
                                          int64_t expected_cols,
                                          int64_t row_offset,
                                          int64_t row_count) -> std::shared_ptr<T>
{
  if (row_count <= 0) { return {}; }
  auto [rows, cols] = read_ibin_shape(filename);
  if (rows != expected_rows || cols != expected_cols) {
    throw std::runtime_error("IBIN shape mismatch for " + filename + ": expected [" +
                             std::to_string(expected_rows) + " x " +
                             std::to_string(expected_cols) + "], got [" +
                             std::to_string(rows) + " x " + std::to_string(cols) + "]");
  }
  if (row_offset < 0 || row_offset + row_count > rows) {
    throw std::runtime_error("IBIN row range out of bounds for " + filename);
  }

  auto values = static_cast<std::size_t>(row_count) * static_cast<std::size_t>(cols);
  auto bytes = values * sizeof(T);
  T* host_storage = nullptr;
  RAFT_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&host_storage), bytes));

  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    cudaFreeHost(host_storage);
    throw std::runtime_error("Cannot open file: " + filename);
  }
  auto payload_offset = static_cast<std::streamoff>(sizeof(int64_t) * 2) +
                        static_cast<std::streamoff>(row_offset * cols * static_cast<int64_t>(sizeof(T)));
  file.seekg(payload_offset, std::ios::beg);
  file.read(reinterpret_cast<char*>(host_storage), bytes);
  if (!file) {
    cudaFreeHost(host_storage);
    throw std::runtime_error("Cannot read IBIN payload from: " + filename);
  }

  return std::shared_ptr<T>(host_storage, [](T* host_ptr) {
    if (host_ptr == nullptr) { return; }
    auto free_ret = cudaFreeHost(host_ptr);
    if (free_ret != cudaSuccess) {
      std::fprintf(stderr,
                   "Warning: cudaFreeHost failed for VecFlow cached host rows: %s\n",
                   cudaGetErrorString(free_ret));
    }
  });
}

template <typename T>
inline auto load_ibin_rows_to_device(shared_resources::configured_raft_resources& res,
                                     const std::string& filename,
                                     int64_t expected_rows,
                                     int64_t expected_cols,
                                     int64_t row_offset,
                                     int64_t row_count) -> std::shared_ptr<T>
{
  auto host_storage =
    load_ibin_rows_to_pinned_host<T>(filename, expected_rows, expected_cols, row_offset, row_count);
  if (host_storage == nullptr) { return {}; }
  return make_device_storage_from_host(
    res,
    host_storage.get(),
    static_cast<std::size_t>(row_count) * static_cast<std::size_t>(expected_cols) * sizeof(T));
}

template <typename T>
class async_host_copy_slot : public std::enable_shared_from_this<async_host_copy_slot<T>> {
 public:
  explicit async_host_copy_slot(int cuda_device_ordinal) : cuda_device_ordinal_(cuda_device_ordinal)
  {
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    stream_created_ = true;
  }

  async_host_copy_slot(const async_host_copy_slot&) = delete;
  auto operator=(const async_host_copy_slot&) -> async_host_copy_slot& = delete;
  async_host_copy_slot(async_host_copy_slot&&) = delete;
  auto operator=(async_host_copy_slot&&) -> async_host_copy_slot& = delete;

  ~async_host_copy_slot() { cleanup(); }

  [[nodiscard]] auto data() const -> T* { return storage_; }
  [[nodiscard]] auto stream() const -> cudaStream_t { return stream_; }
  [[nodiscard]] auto capacity_bytes() const -> std::size_t { return capacity_bytes_; }

  auto try_acquire() -> bool
  {
    auto expected = false;
    return in_use_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
  }

  void release() { in_use_.store(false, std::memory_order_release); }

  [[nodiscard]] auto is_in_use() const -> bool
  {
    return in_use_.load(std::memory_order_acquire);
  }

  void ensure_capacity(std::size_t requested_bytes)
  {
    if (capacity_bytes_ >= requested_bytes && storage_ != nullptr) { return; }
    if (stream_ != nullptr) {
      auto sync_ret = cudaStreamSynchronize(stream_);
      if (sync_ret != cudaSuccess) {
        throw std::runtime_error("cudaStreamSynchronize failed while resizing VecFlow prefetch "
                                 "slot: " +
                                 std::string(cudaGetErrorString(sync_ret)));
      }
    }
    trim_storage();
    RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&storage_), requested_bytes));
    capacity_bytes_ = requested_bytes;
  }

  void trim_storage() noexcept
  {
    if (storage_ == nullptr) { return; }
    if (stream_ != nullptr) { cudaStreamSynchronize(stream_); }
    free_device_storage(cuda_device_ordinal_, storage_);
    storage_        = nullptr;
    capacity_bytes_ = 0;
  }

  auto make_storage_owner() -> std::shared_ptr<T>
  {
    auto self = this->shared_from_this();
    return std::shared_ptr<T>(storage_, [slot = std::move(self)](T*) mutable { slot->release(); });
  }

 private:
  void cleanup() noexcept
  {
    if (stream_ != nullptr) { cudaStreamSynchronize(stream_); }
    trim_storage();
    if (stream_created_ && stream_ != nullptr) {
      auto destroy_ret = cudaStreamDestroy(stream_);
      if (destroy_ret != cudaSuccess) {
        std::fprintf(stderr,
                     "Warning: cudaStreamDestroy failed for VecFlow prefetch stream: %s\n",
                     cudaGetErrorString(destroy_ret));
      }
      stream_         = nullptr;
      stream_created_ = false;
    }
  }

  int cuda_device_ordinal_      = 0;
  T* storage_                   = nullptr;
  std::size_t capacity_bytes_   = 0;
  cudaStream_t stream_          = nullptr;
  bool stream_created_          = false;
  std::atomic<bool> in_use_{false};
};

template <typename T>
class async_host_copy_session : public std::enable_shared_from_this<async_host_copy_session<T>> {
 public:
  explicit async_host_copy_session(int cuda_device_ordinal)
    : cuda_device_ordinal_(cuda_device_ordinal)
  {
  }

  auto acquire_slot(std::size_t requested_bytes) -> std::shared_ptr<async_host_copy_slot<T>>
  {
    trim_idle_slots();
    for (auto const& slot : slots_) {
      if (!slot->try_acquire()) { continue; }
      try {
        slot->ensure_capacity(requested_bytes);
      } catch (...) {
        slot->release();
        throw;
      }
      return slot;
    }

    auto slot = std::make_shared<async_host_copy_slot<T>>(cuda_device_ordinal_);
    if (!slot->try_acquire()) {
      throw std::runtime_error("Failed to acquire a newly created VecFlow prefetch slot");
    }
    try {
      slot->ensure_capacity(requested_bytes);
    } catch (...) {
      slot->release();
      throw;
    }
    slots_.push_back(slot);
    return slot;
  }

 private:
  void trim_idle_slots()
  {
    constexpr std::size_t kMaxIdleSlots = 2;
    std::size_t idle_slots = 0;
    for (auto const& slot : slots_) {
      if (!slot->is_in_use()) { ++idle_slots; }
    }
    if (idle_slots <= kMaxIdleSlots) { return; }

    for (auto it = slots_.begin(); it != slots_.end() && idle_slots > kMaxIdleSlots;) {
      if ((*it)->is_in_use()) {
        ++it;
        continue;
      }
      (*it)->trim_storage();
      it = slots_.erase(it);
      --idle_slots;
    }
  }

  int cuda_device_ordinal_ = 0;
  std::vector<std::shared_ptr<async_host_copy_slot<T>>> slots_;
};

template <typename T>
inline auto acquire_async_host_copy_session() -> std::shared_ptr<async_host_copy_session<T>>
{
  int cuda_device_ordinal = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&cuda_device_ordinal));
  thread_local std::unordered_map<int, std::shared_ptr<async_host_copy_session<T>>> sessions;

  auto session_it = sessions.find(cuda_device_ordinal);
  if (session_it != sessions.end()) { return session_it->second; }

  auto session = std::make_shared<async_host_copy_session<T>>(cuda_device_ordinal);
  sessions[cuda_device_ordinal] = session;
  return session;
}

template <typename T>
class async_host_copy_request {
 public:
  async_host_copy_request(std::shared_ptr<T> host_storage,
                          std::size_t bytes,
                          uint32_t label,
                          int64_t label_offset,
                          int64_t label_size,
                          const char* payload_name = "graph")
    : host_storage_(std::move(host_storage)),
      bytes_(bytes),
      label_(label),
      label_offset_(label_offset),
      label_size_(label_size),
      payload_name_(payload_name == nullptr ? "buffer" : payload_name)
  {
    if (host_storage_ == nullptr) {
      throw std::invalid_argument("Async host graph prefetch requires a valid pinned-host buffer");
    }
    if (bytes_ == 0) { throw std::invalid_argument("Async host graph prefetch requires bytes > 0"); }

    session_ = acquire_async_host_copy_session<T>();
    slot_    = session_->acquire_slot(bytes_);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      slot_->data(), host_storage_.get(), bytes_, cudaMemcpyHostToDevice, slot_->stream()));
    started_ = true;

    vecflow_log("Prefetching cached host ",
                payload_name_,
                " rows [",
                label_offset_,
                ", ",
                (label_offset_ + label_size_),
                ") for label ",
                label_);
  }

  async_host_copy_request(const async_host_copy_request&) = delete;
  auto operator=(const async_host_copy_request&) -> async_host_copy_request& = delete;
  async_host_copy_request(async_host_copy_request&&) = delete;
  auto operator=(async_host_copy_request&&) -> async_host_copy_request& = delete;

  ~async_host_copy_request() { cleanup(); }

  auto wait_and_release() -> std::shared_ptr<T>
  {
    if (started_ && slot_ != nullptr) {
      RAFT_CUDA_TRY(cudaStreamSynchronize(slot_->stream()));
      started_ = false;
    }
    auto device_storage = slot_->make_storage_owner();
    host_storage_.reset();
    slot_.reset();
    session_.reset();
    return device_storage;
  }

 private:
  void cleanup() noexcept
  {
    if (started_ && slot_ != nullptr) { cudaStreamSynchronize(slot_->stream()); }
    host_storage_.reset();
    if (slot_ != nullptr) { slot_->release(); }
    slot_.reset();
    session_.reset();
  }

  std::shared_ptr<T> host_storage_;
  std::size_t bytes_ = 0;
  uint32_t label_ = 0;
  int64_t label_offset_ = 0;
  int64_t label_size_ = 0;
  const char* payload_name_ = "buffer";
  bool started_ = false;
  std::shared_ptr<async_host_copy_session<T>> session_;
  std::shared_ptr<async_host_copy_slot<T>> slot_;
};

using async_host_graph_copy_request = async_host_copy_request<uint32_t>;

template <typename data_t>
inline auto cached_graph_storage_width(const cuvs::neighbors::vecflow::index<data_t>& index) -> int
{
  return index.cagra_graph_storage_width > 0 ? index.cagra_graph_storage_width
                                             : index.cagra_graph_degree;
}

template <typename data_t>
inline auto label_graph_requested_bytes(const cuvs::neighbors::vecflow::index<data_t>& index,
                                        int64_t label_size) -> std::size_t
{
  return static_cast<std::size_t>(label_size) *
         static_cast<std::size_t>(cached_graph_storage_width(index)) * sizeof(uint32_t);
}

template <typename data_t>
auto phoenix_label_graph_tier(cuvs::neighbors::vecflow::index<data_t>& index, uint32_t label) -> int
{
  if (index.phoenix_label_cache == nullptr) { return 2; }
  std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
  if (index.phoenix_label_cache->graphs.find(label) != index.phoenix_label_cache->graphs.end()) {
    return 0;
  }
  if (index.phoenix_label_cache->host_graphs.find(label) !=
      index.phoenix_label_cache->host_graphs.end()) {
    return 1;
  }
  return 2;
}

template <typename data_t>
auto phoenix_label_cache_score_locked(const cuvs::neighbors::vecflow::index<data_t>& index,
                                      uint32_t label,
                                      std::size_t bytes) -> double
{
  if (bytes == 0 || index.phoenix_label_cache == nullptr) { return 0.0; }
  auto accesses = std::uint64_t{0};
  if (label < index.phoenix_label_cache->access_counts.size()) {
    accesses = index.phoenix_label_cache->access_counts[label];
  }
  return static_cast<double>(accesses + 1) / static_cast<double>(bytes);
}

template <typename data_t, typename EntryT>
auto lowest_score_cached_label_locked(cuvs::neighbors::vecflow::index<data_t>& index,
                                      const std::unordered_map<uint32_t, EntryT>& entries)
  -> uint32_t
{
  auto victim_label = std::numeric_limits<uint32_t>::max();
  auto victim_score = std::numeric_limits<double>::infinity();
  for (auto const& [label, entry] : entries) {
    auto score = phoenix_label_cache_score_locked(index, label, entry.bytes);
    if (score < victim_score) {
      victim_score = score;
      victim_label = label;
    }
  }
  return victim_label;
}

template <typename data_t, typename EntryT>
auto sorted_phoenix_labels_by_score_locked(const cuvs::neighbors::vecflow::index<data_t>& index,
                                           const std::unordered_map<uint32_t, EntryT>& entries)
  -> std::vector<uint32_t>
{
  std::vector<std::pair<uint32_t, double>> scored_labels;
  scored_labels.reserve(entries.size());
  for (auto const& [label, entry] : entries) {
    auto score = phoenix_label_cache_score_locked(index, label, entry.bytes);
    scored_labels.emplace_back(label, score);
  }

  std::sort(scored_labels.begin(),
            scored_labels.end(),
            [](auto const& lhs, auto const& rhs) {
              if (lhs.second != rhs.second) { return lhs.second < rhs.second; }
              return lhs.first < rhs.first;
            });

  std::vector<uint32_t> sorted_labels;
  sorted_labels.reserve(scored_labels.size());
  for (auto const& [label, score] : scored_labels) {
    (void)score;
    sorted_labels.push_back(label);
  }
  return sorted_labels;
}

template <typename data_t>
void cache_label_graph_in_hbm(shared_resources::configured_raft_resources& res,
                              cuvs::neighbors::vecflow::index<data_t>& index,
                              uint32_t label,
                              int64_t label_offset,
                              int64_t label_size,
                              std::size_t requested_bytes,
                              const std::shared_ptr<uint32_t>& graph_storage);

template <typename data_t>
void cache_label_graph_in_dram(cuvs::neighbors::vecflow::index<data_t>& index,
                               uint32_t label,
                               int64_t label_offset,
                               int64_t label_size,
                               std::size_t requested_bytes,
                               const std::shared_ptr<uint32_t>& host_graph_storage);

template <typename data_t>
void cache_label_graph_in_dram_locked(cuvs::neighbors::vecflow::index<data_t>& index,
                                      uint32_t label,
                                      int64_t label_offset,
                                      int64_t label_size,
                                      std::size_t requested_bytes,
                                      const std::shared_ptr<uint32_t>& host_graph_storage)
{
  auto cache_capacity = index.phoenix_label_dram_cache_capacity_bytes;
  if (index.phoenix_label_cache == nullptr || cache_capacity == 0 || requested_bytes > cache_capacity ||
      host_graph_storage == nullptr) {
    return;
  }

  auto cache_it = index.phoenix_label_cache->host_graphs.find(label);
  if (cache_it != index.phoenix_label_cache->host_graphs.end()) {
    index.phoenix_label_cache->host_lru_labels.splice(
      index.phoenix_label_cache->host_lru_labels.begin(),
      index.phoenix_label_cache->host_lru_labels,
      cache_it->second.lru_it);
    return;
  }

  while (index.phoenix_label_cache->host_cached_bytes + requested_bytes > cache_capacity &&
         !index.phoenix_label_cache->host_lru_labels.empty()) {
    auto sorted_victims =
      sorted_phoenix_labels_by_score_locked(index, index.phoenix_label_cache->host_graphs);
    auto victim_it = sorted_victims.begin();
    while (index.phoenix_label_cache->host_cached_bytes + requested_bytes > cache_capacity &&
           victim_it != sorted_victims.end()) {
      auto evict_label = *victim_it++;
      auto evict_it = index.phoenix_label_cache->host_graphs.find(evict_label);
      if (evict_it == index.phoenix_label_cache->host_graphs.end()) { continue; }
      vecflow_log("Evicting DRAM graph cache for label ", evict_label, " using score-based policy");
      index.phoenix_label_cache->host_lru_labels.erase(evict_it->second.lru_it);
      index.phoenix_label_cache->host_cached_bytes -= evict_it->second.bytes;
      index.phoenix_label_cache->host_graphs.erase(evict_it);
      index.phoenix_label_cache->dram_evictions += 1;
    }
    break;
  }

  if (index.phoenix_label_cache->host_cached_bytes + requested_bytes > cache_capacity) {
    vecflow_log("DRAM graph cache capacity insufficient after eviction: cached=",
                index.phoenix_label_cache->host_cached_bytes,
                " requested=", requested_bytes, " capacity=", cache_capacity);
    return;
  }

  index.phoenix_label_cache->host_lru_labels.push_front(label);
  index.phoenix_label_cache->host_graphs.emplace(
    label,
    phoenix_label_cache_state::cached_graph_entry{
      host_graph_storage, requested_bytes, index.phoenix_label_cache->host_lru_labels.begin()});
  index.phoenix_label_cache->host_cached_bytes += requested_bytes;
  vecflow_log("Caching graph rows [",
              label_offset,
              ", ",
              (label_offset + label_size),
              ") for label ",
              label,
              " (",
              requested_bytes,
              " bytes in pinned host DRAM)");
}

template <typename data_t>
auto finalize_loaded_phoenix_label_graph(shared_resources::configured_raft_resources& res,
                                         cuvs::neighbors::vecflow::index<data_t>& index,
                                         uint32_t label,
                                         int64_t label_offset,
                                         int64_t label_size,
                                         std::shared_ptr<uint32_t> graph_storage)
  -> std::shared_ptr<uint32_t>
{
  auto requested_bytes = label_graph_requested_bytes(index, label_size);

  if (index.phoenix_label_cache != nullptr && index.phoenix_label_dram_cache_capacity_bytes > 0 &&
      requested_bytes <= index.phoenix_label_dram_cache_capacity_bytes) {
    bool should_populate_dram = false;
    {
      std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
      should_populate_dram = index.phoenix_label_cache->host_graphs.find(label) ==
                             index.phoenix_label_cache->host_graphs.end();
    }
    if (should_populate_dram) {
      auto host_graph_storage =
        make_pinned_host_storage_from_device(res, graph_storage.get(), requested_bytes);
      cache_label_graph_in_dram(
        index, label, label_offset, label_size, requested_bytes, host_graph_storage);
    }
  }

  cache_label_graph_in_hbm(
    res, index, label, label_offset, label_size, requested_bytes, graph_storage);
  return graph_storage;
}

template <typename data_t>
void cache_label_graph_in_hbm(shared_resources::configured_raft_resources& res,
                              cuvs::neighbors::vecflow::index<data_t>& index,
                              uint32_t label,
                              int64_t label_offset,
                              int64_t label_size,
                              std::size_t requested_bytes,
                              const std::shared_ptr<uint32_t>& graph_storage)
{
  auto cache_capacity = index.phoenix_label_cache_capacity_bytes;
  if (index.phoenix_label_cache == nullptr || cache_capacity == 0 || requested_bytes > cache_capacity) {
    return;
  }

  std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
  auto cache_it = index.phoenix_label_cache->graphs.find(label);
  if (cache_it != index.phoenix_label_cache->graphs.end()) {
    index.phoenix_label_cache->lru_labels.splice(index.phoenix_label_cache->lru_labels.begin(),
                                                 index.phoenix_label_cache->lru_labels,
                                                 cache_it->second.lru_it);
    return;
  }

  while (index.phoenix_label_cache->cached_bytes + requested_bytes > cache_capacity &&
         !index.phoenix_label_cache->lru_labels.empty()) {
    auto sorted_victims =
      sorted_phoenix_labels_by_score_locked(index, index.phoenix_label_cache->graphs);
    auto victim_it = sorted_victims.begin();
    while (index.phoenix_label_cache->cached_bytes + requested_bytes > cache_capacity &&
           victim_it != sorted_victims.end()) {
      auto evict_label = *victim_it++;
      auto evict_it = index.phoenix_label_cache->graphs.find(evict_label);
      if (evict_it == index.phoenix_label_cache->graphs.end()) { continue; }
      auto evicted_storage = evict_it->second.storage;
      auto evicted_bytes = evict_it->second.bytes;
      vecflow_log("Evicting HBM graph cache for label ", evict_label, " using score-based policy");
      index.phoenix_label_cache->lru_labels.erase(evict_it->second.lru_it);
      index.phoenix_label_cache->cached_bytes -= evicted_bytes;
      index.phoenix_label_cache->graphs.erase(evict_it);
      index.phoenix_label_cache->hbm_evictions += 1;
      if (detail::phoenix::cascade_eviction_enabled() &&
          index.phoenix_label_dram_cache_capacity_bytes > 0 &&
          evict_label < index.host_cagra_label_size.size()) {
        auto evict_label_size = static_cast<int64_t>(index.host_cagra_label_size[evict_label]);
        auto evict_label_offset = static_cast<int64_t>(index.host_cagra_label_offset[evict_label]);
        if (evict_label_size > 0 && evicted_storage != nullptr) {
          auto host_graph_storage =
            make_pinned_host_storage_from_device(res, evicted_storage.get(), evicted_bytes);
          cache_label_graph_in_dram_locked(
            index, evict_label, evict_label_offset, evict_label_size, evicted_bytes, host_graph_storage);
        }
      }
    }
    break;
  }

  if (index.phoenix_label_cache->cached_bytes + requested_bytes > cache_capacity) {
    vecflow_log("HBM graph cache capacity insufficient after eviction: cached=",
                index.phoenix_label_cache->cached_bytes,
                " requested=", requested_bytes, " capacity=", cache_capacity);
    return;
  }

  index.phoenix_label_cache->lru_labels.push_front(label);
  index.phoenix_label_cache->graphs.emplace(
    label,
    phoenix_label_cache_state::cached_graph_entry{
      graph_storage, requested_bytes, index.phoenix_label_cache->lru_labels.begin()});
  index.phoenix_label_cache->cached_bytes += requested_bytes;
  vecflow_log("Caching graph rows [",
              label_offset,
              ", ",
              (label_offset + label_size),
              ") for label ",
              label,
              " (",
              requested_bytes,
              " bytes in HBM)");
}

template <typename data_t>
void cache_label_graph_in_dram(cuvs::neighbors::vecflow::index<data_t>& index,
                               uint32_t label,
                               int64_t label_offset,
                               int64_t label_size,
                               std::size_t requested_bytes,
                               const std::shared_ptr<uint32_t>& host_graph_storage)
{
  if (index.phoenix_label_cache == nullptr) { return; }
  std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
  cache_label_graph_in_dram_locked(
    index, label, label_offset, label_size, requested_bytes, host_graph_storage);
}

template <typename data_t>
inline auto label_dataset_requested_bytes(int64_t label_size, int64_t dim) -> std::size_t
{
  return static_cast<std::size_t>(label_size) * static_cast<std::size_t>(dim) * sizeof(data_t);
}

template <typename data_t>
auto phoenix_label_dataset_tier(cuvs::neighbors::vecflow::index<data_t>& index, uint32_t label)
  -> int
{
  if (index.phoenix_label_dataset_cache == nullptr) { return 2; }
  std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
  if (index.phoenix_label_dataset_cache->datasets.find(label) !=
      index.phoenix_label_dataset_cache->datasets.end()) {
    return 0;
  }
  if (index.phoenix_label_dataset_cache->host_datasets.find(label) !=
      index.phoenix_label_dataset_cache->host_datasets.end()) {
    return 1;
  }
  return 2;
}

template <typename data_t>
void cache_label_dataset_in_hbm(shared_resources::configured_raft_resources& res,
                                cuvs::neighbors::vecflow::index<data_t>& index,
                                uint32_t label,
                                int64_t label_offset,
                                int64_t label_size,
                                std::size_t requested_bytes,
                                const std::shared_ptr<data_t>& dataset_storage);

template <typename data_t>
void cache_label_dataset_in_dram(cuvs::neighbors::vecflow::index<data_t>& index,
                                 uint32_t label,
                                 int64_t label_offset,
                                 int64_t label_size,
                                 std::size_t requested_bytes,
                                 const std::shared_ptr<data_t>& host_dataset_storage);

template <typename data_t>
void cache_label_dataset_in_dram_locked(cuvs::neighbors::vecflow::index<data_t>& index,
                                        uint32_t label,
                                        int64_t label_offset,
                                        int64_t label_size,
                                        std::size_t requested_bytes,
                                        const std::shared_ptr<data_t>& host_dataset_storage)
{
  auto cache_capacity = index.phoenix_label_dataset_dram_cache_capacity_bytes;
  if (index.phoenix_label_dataset_cache == nullptr || cache_capacity == 0 ||
      requested_bytes > cache_capacity || host_dataset_storage == nullptr) {
    return;
  }

  auto cache_it = index.phoenix_label_dataset_cache->host_datasets.find(label);
  if (cache_it != index.phoenix_label_dataset_cache->host_datasets.end()) {
    index.phoenix_label_dataset_cache->host_lru_labels.splice(
      index.phoenix_label_dataset_cache->host_lru_labels.begin(),
      index.phoenix_label_dataset_cache->host_lru_labels,
      cache_it->second.lru_it);
    return;
  }

  while (index.phoenix_label_dataset_cache->host_cached_bytes + requested_bytes > cache_capacity &&
         !index.phoenix_label_dataset_cache->host_lru_labels.empty()) {
    auto sorted_victims =
      sorted_phoenix_labels_by_score_locked(index, index.phoenix_label_dataset_cache->host_datasets);
    auto victim_it = sorted_victims.begin();
    while (index.phoenix_label_dataset_cache->host_cached_bytes + requested_bytes > cache_capacity &&
           victim_it != sorted_victims.end()) {
      auto evict_label = *victim_it++;
      auto evict_it = index.phoenix_label_dataset_cache->host_datasets.find(evict_label);
      if (evict_it == index.phoenix_label_dataset_cache->host_datasets.end()) { continue; }
      vecflow_log("Evicting DRAM dataset cache for label ", evict_label, " using score-based policy");
      index.phoenix_label_dataset_cache->host_lru_labels.erase(evict_it->second.lru_it);
      index.phoenix_label_dataset_cache->host_cached_bytes -= evict_it->second.bytes;
      index.phoenix_label_dataset_cache->host_datasets.erase(evict_it);
      index.phoenix_label_dataset_cache->dram_evictions += 1;
    }
    break;
  }

  if (index.phoenix_label_dataset_cache->host_cached_bytes + requested_bytes > cache_capacity) {
    vecflow_log("DRAM dataset cache capacity insufficient after eviction: cached=",
                index.phoenix_label_dataset_cache->host_cached_bytes,
                " requested=", requested_bytes, " capacity=", cache_capacity);
    return;
  }

  index.phoenix_label_dataset_cache->host_lru_labels.push_front(label);
  index.phoenix_label_dataset_cache->host_datasets.emplace(
    label,
    typename phoenix_label_dataset_cache_state<data_t>::cached_dataset_entry{
      host_dataset_storage,
      requested_bytes,
      index.phoenix_label_dataset_cache->host_lru_labels.begin()});
  index.phoenix_label_dataset_cache->host_cached_bytes += requested_bytes;
  vecflow_log("Caching dataset rows [",
              label_offset,
              ", ",
              (label_offset + label_size),
              ") for label ",
              label,
              " (",
              requested_bytes,
              " bytes in pinned host DRAM)");
}

template <typename data_t>
auto finalize_loaded_phoenix_label_dataset(shared_resources::configured_raft_resources& res,
                                           cuvs::neighbors::vecflow::index<data_t>& index,
                                           uint32_t label,
                                           int64_t label_offset,
                                           int64_t label_size,
                                           int64_t dim,
                                           std::shared_ptr<data_t> dataset_storage)
  -> std::shared_ptr<data_t>
{
  auto requested_bytes = label_dataset_requested_bytes<data_t>(label_size, dim);
  if (index.phoenix_label_dataset_cache != nullptr &&
      index.phoenix_label_dataset_dram_cache_capacity_bytes > 0 &&
      requested_bytes <= index.phoenix_label_dataset_dram_cache_capacity_bytes) {
    bool should_populate_dram = false;
    {
      std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
      should_populate_dram = index.phoenix_label_dataset_cache->host_datasets.find(label) ==
                             index.phoenix_label_dataset_cache->host_datasets.end();
    }
    if (should_populate_dram) {
      auto host_dataset_storage =
        make_pinned_host_storage_from_device(res, dataset_storage.get(), requested_bytes);
      cache_label_dataset_in_dram(
        index, label, label_offset, label_size, requested_bytes, host_dataset_storage);
    }
  }

  cache_label_dataset_in_hbm(
    res, index, label, label_offset, label_size, requested_bytes, dataset_storage);
  return dataset_storage;
}

template <typename data_t>
void cache_label_dataset_in_hbm(shared_resources::configured_raft_resources& res,
                                cuvs::neighbors::vecflow::index<data_t>& index,
                                uint32_t label,
                                int64_t label_offset,
                                int64_t label_size,
                                std::size_t requested_bytes,
                                const std::shared_ptr<data_t>& dataset_storage)
{
  auto cache_capacity = index.phoenix_label_dataset_cache_capacity_bytes;
  if (index.phoenix_label_dataset_cache == nullptr || cache_capacity == 0 ||
      requested_bytes > cache_capacity) {
    return;
  }

  std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
  auto cache_it = index.phoenix_label_dataset_cache->datasets.find(label);
  if (cache_it != index.phoenix_label_dataset_cache->datasets.end()) {
    index.phoenix_label_dataset_cache->lru_labels.splice(
      index.phoenix_label_dataset_cache->lru_labels.begin(),
      index.phoenix_label_dataset_cache->lru_labels,
      cache_it->second.lru_it);
    return;
  }

  while (index.phoenix_label_dataset_cache->cached_bytes + requested_bytes > cache_capacity &&
         !index.phoenix_label_dataset_cache->lru_labels.empty()) {
    auto sorted_victims =
      sorted_phoenix_labels_by_score_locked(index, index.phoenix_label_dataset_cache->datasets);
    auto victim_it = sorted_victims.begin();
    while (index.phoenix_label_dataset_cache->cached_bytes + requested_bytes > cache_capacity &&
           victim_it != sorted_victims.end()) {
      auto evict_label = *victim_it++;
      auto evict_it = index.phoenix_label_dataset_cache->datasets.find(evict_label);
      if (evict_it == index.phoenix_label_dataset_cache->datasets.end()) { continue; }
      auto evicted_storage = evict_it->second.storage;
      auto evicted_bytes = evict_it->second.bytes;
      vecflow_log("Evicting HBM dataset cache for label ", evict_label, " using score-based policy");
      index.phoenix_label_dataset_cache->lru_labels.erase(evict_it->second.lru_it);
      index.phoenix_label_dataset_cache->cached_bytes -= evicted_bytes;
      index.phoenix_label_dataset_cache->datasets.erase(evict_it);
      index.phoenix_label_dataset_cache->hbm_evictions += 1;
      if (detail::phoenix::cascade_eviction_enabled() &&
          index.phoenix_label_dataset_dram_cache_capacity_bytes > 0 &&
          evict_label < index.host_cagra_label_size.size()) {
        auto evict_label_size = static_cast<int64_t>(index.host_cagra_label_size[evict_label]);
        auto evict_label_offset = static_cast<int64_t>(index.host_cagra_label_offset[evict_label]);
        if (evict_label_size > 0 && evicted_storage != nullptr) {
          auto host_dataset_storage =
            make_pinned_host_storage_from_device(res, evicted_storage.get(), evicted_bytes);
          cache_label_dataset_in_dram_locked(index,
                                             evict_label,
                                             evict_label_offset,
                                             evict_label_size,
                                             evicted_bytes,
                                             host_dataset_storage);
        }
      }
    }
    break;
  }

  if (index.phoenix_label_dataset_cache->cached_bytes + requested_bytes > cache_capacity) {
    vecflow_log("HBM dataset cache capacity insufficient after eviction: cached=",
                index.phoenix_label_dataset_cache->cached_bytes,
                " requested=", requested_bytes, " capacity=", cache_capacity);
    return;
  }

  index.phoenix_label_dataset_cache->lru_labels.push_front(label);
  index.phoenix_label_dataset_cache->datasets.emplace(
    label,
    typename phoenix_label_dataset_cache_state<data_t>::cached_dataset_entry{
      dataset_storage, requested_bytes, index.phoenix_label_dataset_cache->lru_labels.begin()});
  index.phoenix_label_dataset_cache->cached_bytes += requested_bytes;
  vecflow_log("Caching dataset rows [",
              label_offset,
              ", ",
              (label_offset + label_size),
              ") for label ",
              label,
              " (",
              requested_bytes,
              " bytes in HBM)");
}

template <typename data_t>
void cache_label_dataset_in_dram(cuvs::neighbors::vecflow::index<data_t>& index,
                                 uint32_t label,
                                 int64_t label_offset,
                                 int64_t label_size,
                                 std::size_t requested_bytes,
                                 const std::shared_ptr<data_t>& host_dataset_storage)
{
  if (index.phoenix_label_dataset_cache == nullptr) { return; }
  std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
  cache_label_dataset_in_dram_locked(
    index, label, label_offset, label_size, requested_bytes, host_dataset_storage);
}

template <typename data_t>
inline auto bfs_label_dram_requested_bytes(int64_t label_size, int64_t dim) -> std::size_t
{
  return static_cast<std::size_t>(label_size) * static_cast<std::size_t>(dim) * sizeof(data_t);
}

template <typename data_t>
inline auto bfs_label_hbm_requested_bytes(int64_t label_size, int64_t dim) -> std::size_t
{
  auto padded_size =
    raft::ceildiv<std::size_t>(static_cast<std::size_t>(label_size),
                               static_cast<std::size_t>(cuvs::neighbors::ivf_flat::kIndexGroupSize)) *
    static_cast<std::size_t>(cuvs::neighbors::ivf_flat::kIndexGroupSize);
  return padded_size * static_cast<std::size_t>(dim) * sizeof(data_t) +
         padded_size * sizeof(int64_t);
}

template <typename data_t>
auto bfs_label_tier(cuvs::neighbors::vecflow::index<data_t>& index, uint32_t label) -> int
{
  if (index.bfs_cache == nullptr) { return 2; }
  {
    std::lock_guard<std::mutex> lock(index.bfs_cache->hbm_mutex);
    if (index.bfs_cache->hbm_entries.find(label) != index.bfs_cache->hbm_entries.end()) { return 0; }
  }
  {
    std::lock_guard<std::mutex> lock(index.bfs_cache->dram_mutex);
    if (index.bfs_cache->dram_entries.find(label) != index.bfs_cache->dram_entries.end()) { return 1; }
  }
  return 2;
}

template <typename data_t>
auto bfs_label_cache_score_locked(const cuvs::neighbors::vecflow::index<data_t>& index,
                                  uint32_t label,
                                  std::size_t bytes) -> double
{
  if (bytes == 0 || index.bfs_cache == nullptr) { return 0.0; }
  auto accesses = std::uint64_t{0};
  {
    std::lock_guard<std::mutex> access_lock(index.bfs_cache->access_mutex);
    if (label < index.bfs_cache->access_counts.size()) {
      accesses = index.bfs_cache->access_counts[label];
    }
  }
  return static_cast<double>(accesses + 1) / static_cast<double>(bytes);
}

template <typename data_t, typename EntryT>
auto lowest_score_bfs_label_locked(cuvs::neighbors::vecflow::index<data_t>& index,
                                   const std::unordered_map<uint32_t, EntryT>& entries) -> uint32_t
{
  auto victim_label = std::numeric_limits<uint32_t>::max();
  auto victim_score = std::numeric_limits<double>::infinity();
  for (auto const& [label, entry] : entries) {
    auto score = bfs_label_cache_score_locked(index, label, entry.bytes);
    if (score < victim_score) {
      victim_score = score;
      victim_label = label;
    }
  }
  return victim_label;
}

template <typename data_t, typename EntryT>
auto sorted_bfs_labels_by_score_locked(cuvs::neighbors::vecflow::index<data_t>& index,
                                       const std::unordered_map<uint32_t, EntryT>& entries)
  -> std::vector<uint32_t>
{
  std::vector<std::uint64_t> access_snapshot;
  if (index.bfs_cache != nullptr) {
    std::lock_guard<std::mutex> access_lock(index.bfs_cache->access_mutex);
    access_snapshot = index.bfs_cache->access_counts;
  }

  std::vector<std::pair<uint32_t, double>> scored_labels;
  scored_labels.reserve(entries.size());
  for (auto const& [label, entry] : entries) {
    auto accesses = label < access_snapshot.size() ? access_snapshot[label] : std::uint64_t{0};
    auto score = entry.bytes == 0
                   ? 0.0
                   : static_cast<double>(accesses + 1) / static_cast<double>(entry.bytes);
    scored_labels.emplace_back(label, score);
  }

  std::sort(scored_labels.begin(),
            scored_labels.end(),
            [](auto const& lhs, auto const& rhs) {
              if (lhs.second != rhs.second) { return lhs.second < rhs.second; }
              return lhs.first < rhs.first;
            });

  std::vector<uint32_t> sorted_labels;
  sorted_labels.reserve(scored_labels.size());
  for (auto const& [label, score] : scored_labels) {
    (void)score;
    sorted_labels.push_back(label);
  }
  return sorted_labels;
}

template <typename data_t>
auto build_bfs_label_index_from_device_rows(shared_resources::configured_raft_resources& res,
                                            raft::device_matrix_view<const data_t, int64_t> rows,
                                            const uint32_t* host_index_map)
  -> std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>>
{
  auto row_count = rows.extent(0);
  if (row_count <= 0) { return {}; }

  auto label_index =
    std::make_shared<cuvs::neighbors::ivf_flat::index<data_t, int64_t>>(res);
  auto d_index_map = raft::make_device_vector<uint32_t, int64_t>(res, row_count);
  auto d_label_size = raft::make_device_vector<uint32_t, int64_t>(res, 1);
  auto d_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, 1);
  auto stream = raft::resource::get_cuda_stream(res);
  auto row_count_us = static_cast<std::size_t>(row_count);
  uint32_t host_label_size_u32 = static_cast<uint32_t>(row_count);
  uint32_t host_label_offset_u32 = 0;
  std::vector<uint32_t> local_index_map(row_count_us);
  std::iota(local_index_map.begin(), local_index_map.end(), uint32_t{0});
  raft::update_device(d_index_map.data_handle(), local_index_map.data(), row_count, stream);
  raft::update_device(d_label_size.data_handle(), &host_label_size_u32, 1, stream);
  raft::update_device(d_label_offset.data_handle(), &host_label_offset_u32, 1, stream);
  build_filtered_bfs(res,
                     label_index.get(),
                     raft::make_const_mdspan(rows),
                     d_index_map.view(),
                     d_label_size.view(),
                     d_label_offset.view());
  raft::resource::sync_stream(res);

  std::vector<int64_t> global_index_map(row_count_us);
  std::transform(host_index_map,
                 host_index_map + row_count,
                 global_index_map.begin(),
                 [](uint32_t row_id) { return static_cast<int64_t>(row_id); });
  auto list = label_index->lists()[0];
  if (list == nullptr) { throw std::runtime_error("BFS label index list is unexpectedly null"); }
  raft::update_device(
    list->indices.data_handle(), global_index_map.data(), static_cast<int64_t>(row_count_us), stream);
  cuvs::neighbors::ivf_flat::helpers::recompute_internal_state(res, label_index.get());
  raft::resource::sync_stream(res);
  return label_index;
}

template <typename data_t>
void cache_bfs_label_in_dram_locked(cuvs::neighbors::vecflow::index<data_t>& index,
                                    uint32_t label,
                                    int64_t label_size,
                                    std::size_t requested_bytes,
                                    const std::shared_ptr<data_t>& host_storage)
{
  auto cache_capacity = index.bfs_dram_capacity_bytes;
  if (index.bfs_cache == nullptr || cache_capacity == 0 || requested_bytes > cache_capacity ||
      host_storage == nullptr) {
    return;
  }

  auto cache_it = index.bfs_cache->dram_entries.find(label);
  if (cache_it != index.bfs_cache->dram_entries.end()) {
    index.bfs_cache->dram_lru.splice(
      index.bfs_cache->dram_lru.begin(), index.bfs_cache->dram_lru, cache_it->second.lru_it);
    return;
  }

  if (index.bfs_cache->dram_cached_bytes + requested_bytes > cache_capacity) {
    auto sorted_victims = sorted_bfs_labels_by_score_locked(index, index.bfs_cache->dram_entries);
    auto victim_it = sorted_victims.begin();
    while (index.bfs_cache->dram_cached_bytes + requested_bytes > cache_capacity &&
           victim_it != sorted_victims.end() && !index.bfs_cache->dram_lru.empty()) {
      auto evict_label = *victim_it++;
      auto evict_it = index.bfs_cache->dram_entries.find(evict_label);
      if (evict_it == index.bfs_cache->dram_entries.end()) { continue; }
      vecflow_log("Evicting DRAM BFS cache for label ", evict_label, " using score-based policy");
      index.bfs_cache->dram_lru.erase(evict_it->second.lru_it);
      index.bfs_cache->dram_cached_bytes -= evict_it->second.bytes;
      index.bfs_cache->dram_entries.erase(evict_it);
      index.bfs_cache->dram_evictions += 1;
    }
  }

  if (index.bfs_cache->dram_cached_bytes + requested_bytes > cache_capacity) { return; }

  index.bfs_cache->dram_lru.push_front(label);
  index.bfs_cache->dram_entries.emplace(
    label,
    typename bfs_label_cache_state<data_t>::cached_bfs_entry{
      nullptr, host_storage, requested_bytes, label_size, index.bfs_cache->dram_lru.begin()});
  index.bfs_cache->dram_cached_bytes += requested_bytes;
}

template <typename data_t>
void cache_bfs_label_in_dram(cuvs::neighbors::vecflow::index<data_t>& index,
                             uint32_t label,
                             int64_t label_size,
                             std::size_t requested_bytes,
                             const std::shared_ptr<data_t>& host_storage)
{
  if (index.bfs_cache == nullptr) { return; }
  std::lock_guard<std::mutex> lock(index.bfs_cache->dram_mutex);
  cache_bfs_label_in_dram_locked(index, label, label_size, requested_bytes, host_storage);
}

template <typename data_t>
void cache_bfs_label_in_hbm(shared_resources::configured_raft_resources& res,
                            cuvs::neighbors::vecflow::index<data_t>& index,
                            uint32_t label,
                            int64_t label_size,
                            std::size_t requested_bytes,
                            const std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>>&
                              label_index)
{
  auto cache_capacity = index.bfs_hbm_capacity_bytes;
  if (index.bfs_cache == nullptr || cache_capacity == 0 || requested_bytes > cache_capacity ||
      label_index == nullptr) {
    return;
  }

  struct hbm_eviction_candidate {
    uint32_t label = std::numeric_limits<uint32_t>::max();
    int64_t label_size = 0;
    std::size_t dram_bytes = 0;
    std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>> index;
  };
  std::vector<hbm_eviction_candidate> eviction_candidates;

  {
    std::lock_guard<std::mutex> lock(index.bfs_cache->hbm_mutex);
    auto cache_it = index.bfs_cache->hbm_entries.find(label);
    if (cache_it != index.bfs_cache->hbm_entries.end()) {
      index.bfs_cache->hbm_lru.splice(
        index.bfs_cache->hbm_lru.begin(), index.bfs_cache->hbm_lru, cache_it->second.lru_it);
      return;
    }

    if (index.bfs_cache->hbm_cached_bytes + requested_bytes > cache_capacity) {
      auto sorted_victims = sorted_bfs_labels_by_score_locked(index, index.bfs_cache->hbm_entries);
      auto victim_it = sorted_victims.begin();
      while (index.bfs_cache->hbm_cached_bytes + requested_bytes > cache_capacity &&
             victim_it != sorted_victims.end()) {
        auto evict_label = *victim_it++;
        auto evict_it = index.bfs_cache->hbm_entries.find(evict_label);
        if (evict_it == index.bfs_cache->hbm_entries.end()) { continue; }

        vecflow_log("Evicting HBM BFS cache for label ", evict_label, " using score-based policy");
        auto evict_hbm_bytes = evict_it->second.bytes;
        auto should_cascade = detail::phoenix::cascade_eviction_enabled() &&
                              index.bfs_dram_capacity_bytes > 0 && evict_it->second.index != nullptr &&
                              evict_it->second.label_size > 0;
        if (should_cascade) {
          eviction_candidates.push_back(hbm_eviction_candidate{evict_label,
                                                               evict_it->second.label_size,
                                                               bfs_label_dram_requested_bytes<data_t>(
                                                                 evict_it->second.label_size,
                                                                 index.dataset_dim),
                                                               evict_it->second.index});
        }
        index.bfs_cache->hbm_lru.erase(evict_it->second.lru_it);
        index.bfs_cache->hbm_cached_bytes -= evict_hbm_bytes;
        index.bfs_cache->hbm_entries.erase(evict_it);
        index.bfs_cache->hbm_evictions += 1;
      }

      if (index.bfs_cache->hbm_cached_bytes + requested_bytes > cache_capacity) {
        vecflow_log("BFS HBM cache capacity insufficient after eviction: cached=",
                    index.bfs_cache->hbm_cached_bytes,
                    " requested=", requested_bytes, " capacity=", cache_capacity);
        return;
      }
    }
  }

  struct dram_restore_candidate {
    uint32_t label = std::numeric_limits<uint32_t>::max();
    int64_t label_size = 0;
    std::size_t requested_bytes = 0;
    std::shared_ptr<data_t> host_storage;
  };
  std::vector<dram_restore_candidate> dram_restores;
  dram_restores.reserve(eviction_candidates.size());

  for (auto& evicted : eviction_candidates) {
    if (evicted.index == nullptr || evicted.label_size <= 0 || evicted.dram_bytes == 0) {
      continue;
    }
    auto row_count_u32 = static_cast<uint32_t>(evicted.label_size);
    auto dim_u32 = static_cast<uint32_t>(index.dataset_dim);
    auto unpacked_rows = raft::make_device_matrix<data_t, uint32_t>(res, row_count_u32, dim_u32);
    auto list = evicted.index->lists()[0];
    if (list == nullptr) { continue; }

    cuvs::neighbors::ivf_flat::helpers::codepacker::unpack(
      res,
      list->data.view(),
      evicted.index->veclen(),
      0,
      raft::make_device_matrix_view<data_t, uint32_t, raft::row_major>(
        unpacked_rows.data_handle(), row_count_u32, dim_u32));
    raft::resource::sync_stream(res);
    dram_restores.push_back(dram_restore_candidate{evicted.label,
                                                   evicted.label_size,
                                                   evicted.dram_bytes,
                                                   make_pinned_host_storage_from_device(
                                                     res,
                                                     unpacked_rows.data_handle(),
                                                     evicted.dram_bytes)});
  }

  if (!dram_restores.empty()) {
    std::lock_guard<std::mutex> lock(index.bfs_cache->dram_mutex);
    for (auto& restore : dram_restores) {
      cache_bfs_label_in_dram_locked(
        index, restore.label, restore.label_size, restore.requested_bytes, restore.host_storage);
    }
  }

  std::lock_guard<std::mutex> lock(index.bfs_cache->hbm_mutex);
  auto cache_it = index.bfs_cache->hbm_entries.find(label);
  if (cache_it != index.bfs_cache->hbm_entries.end()) {
    index.bfs_cache->hbm_lru.splice(
      index.bfs_cache->hbm_lru.begin(), index.bfs_cache->hbm_lru, cache_it->second.lru_it);
    return;
  }
  if (index.bfs_cache->hbm_cached_bytes + requested_bytes > cache_capacity) {
    vecflow_log("BFS HBM cache capacity insufficient on re-check: cached=",
                index.bfs_cache->hbm_cached_bytes,
                " requested=", requested_bytes, " capacity=", cache_capacity);
    return;
  }

  index.bfs_cache->hbm_lru.push_front(label);
  index.bfs_cache->hbm_entries.emplace(
    label,
    typename bfs_label_cache_state<data_t>::cached_bfs_entry{
      label_index, nullptr, requested_bytes, label_size, index.bfs_cache->hbm_lru.begin()});
  index.bfs_cache->hbm_cached_bytes += requested_bytes;
}

template <typename data_t>
auto finalize_loaded_bfs_label(shared_resources::configured_raft_resources& res,
                               cuvs::neighbors::vecflow::index<data_t>& index,
                               uint32_t label,
                               int64_t label_offset,
                               int64_t label_size,
                               std::shared_ptr<data_t> device_rows,
                               bool populate_dram_cache = true)
  -> std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>>
{
  auto dram_bytes = bfs_label_dram_requested_bytes<data_t>(label_size, index.dataset_dim);
  auto hbm_bytes = bfs_label_hbm_requested_bytes<data_t>(label_size, index.dataset_dim);

  if (populate_dram_cache && index.bfs_cache != nullptr && index.bfs_dram_capacity_bytes > 0 &&
      dram_bytes <= index.bfs_dram_capacity_bytes) {
    bool should_populate_dram = false;
    {
      std::lock_guard<std::mutex> lock(index.bfs_cache->dram_mutex);
      should_populate_dram = index.bfs_cache->dram_entries.find(label) == index.bfs_cache->dram_entries.end();
    }
    if (should_populate_dram && device_rows != nullptr) {
      auto host_storage = make_pinned_host_storage_from_device(res, device_rows.get(), dram_bytes);
      cache_bfs_label_in_dram(index, label, label_size, dram_bytes, host_storage);
    }
  }

  if (device_rows == nullptr) { return {}; }
  auto rows_view = raft::make_device_matrix_view<const data_t, int64_t, raft::row_major>(
    device_rows.get(), label_size, index.dataset_dim);
  auto label_index =
    build_bfs_label_index_from_device_rows(res, rows_view, index.host_bfs_index_map.data() + label_offset);
  cache_bfs_label_in_hbm(res, index, label, label_size, hbm_bytes, label_index);
  return label_index;
}

template <typename data_t>
auto get_or_load_bfs_label_index(shared_resources::configured_raft_resources& res,
                                 cuvs::neighbors::vecflow::index<data_t>& index,
                                 uint32_t label,
                                 reusable_device_buffer<data_t>* reusable_buffer = nullptr)
  -> std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>>
{
  if (label >= index.host_bfs_label_size.size() || label >= index.host_bfs_label_offset.size()) {
    throw std::runtime_error("BFS cache label is out of bounds: " + std::to_string(label));
  }
  if (index.bfs_dataset_cache_fname.empty()) {
    throw std::runtime_error("Tiered BFS search requires a packed BFS dataset cache file.");
  }

  auto label_size = static_cast<int64_t>(index.host_bfs_label_size[label]);
  auto label_offset = static_cast<int64_t>(index.host_bfs_label_offset[label]);
  if (label_size <= 0) {
    throw std::runtime_error("Tiered BFS search received a non-BFS label " + std::to_string(label));
  }

  if (index.bfs_cache != nullptr && index.bfs_hbm_capacity_bytes > 0) {
    std::lock_guard<std::mutex> lock(index.bfs_cache->hbm_mutex);
    auto cache_it = index.bfs_cache->hbm_entries.find(label);
    if (cache_it != index.bfs_cache->hbm_entries.end()) {
      index.bfs_cache->hbm_lru.splice(
        index.bfs_cache->hbm_lru.begin(), index.bfs_cache->hbm_lru, cache_it->second.lru_it);
      index.bfs_cache->hbm_hits += 1;
      return cache_it->second.index;
    }
  }

  auto dram_bytes = bfs_label_dram_requested_bytes<data_t>(label_size, index.dataset_dim);
  if (index.bfs_cache != nullptr && index.bfs_dram_capacity_bytes > 0) {
    std::shared_ptr<data_t> host_storage;
    {
      std::lock_guard<std::mutex> lock(index.bfs_cache->dram_mutex);
      auto cache_it = index.bfs_cache->dram_entries.find(label);
      if (cache_it != index.bfs_cache->dram_entries.end()) {
        index.bfs_cache->dram_lru.splice(
          index.bfs_cache->dram_lru.begin(), index.bfs_cache->dram_lru, cache_it->second.lru_it);
        host_storage = cache_it->second.storage;
        index.bfs_cache->dram_hits += 1;
      }
    }
    if (host_storage != nullptr) {
      auto device_rows = reusable_buffer != nullptr
        ? reusable_buffer->copy_from_host(res, host_storage.get(), dram_bytes)
        : make_device_storage_from_host(res, host_storage.get(), dram_bytes);
      return finalize_loaded_bfs_label(
        res, index, label, label_offset, label_size, std::move(device_rows), false);
    }
  }

  if (index.bfs_cache != nullptr) {
    std::lock_guard<std::mutex> lock(index.bfs_cache->access_mutex);
    index.bfs_cache->ssd_loads += 1;
  }
  auto device_rows = load_ibin_rows_to_device<data_t>(
    res, index.bfs_dataset_cache_fname, index.bfs_total_rows, index.dataset_dim, label_offset, label_size);
  return finalize_loaded_bfs_label(
    res, index, label, label_offset, label_size, std::move(device_rows));
}

template <typename data_t>
void rebalance_bfs_label_cache(shared_resources::configured_raft_resources& res,
                               cuvs::neighbors::vecflow::index<data_t>& index)
{
  if (index.bfs_cache == nullptr || index.bfs_hbm_capacity_bytes == 0 ||
      index.bfs_dram_capacity_bytes == 0) {
    return;
  }

  std::shared_ptr<data_t> host_storage;
  uint32_t promote_label = std::numeric_limits<uint32_t>::max();
  double best_dram_score = 0.0;
  double worst_hbm_score = std::numeric_limits<double>::infinity();
  bool has_hbm_entries = false;

  {
    std::lock_guard<std::mutex> lock(index.bfs_cache->hbm_mutex);
    for (auto const& [label, entry] : index.bfs_cache->hbm_entries) {
      has_hbm_entries = true;
      worst_hbm_score =
        std::min(worst_hbm_score, bfs_label_cache_score_locked(index, label, entry.bytes));
    }
  }
  {
    std::lock_guard<std::mutex> lock(index.bfs_cache->dram_mutex);
    for (auto const& [label, entry] : index.bfs_cache->dram_entries) {
      auto score = bfs_label_cache_score_locked(index, label, entry.bytes);
      if (score > best_dram_score) {
        best_dram_score = score;
        promote_label = label;
        host_storage = entry.storage;
      }
    }
  }

  if (promote_label == std::numeric_limits<uint32_t>::max() || host_storage == nullptr ||
      (has_hbm_entries && best_dram_score <= worst_hbm_score) ||
      promote_label >= index.host_bfs_label_size.size() ||
      promote_label >= index.host_bfs_label_offset.size()) {
    return;
  }

  auto label_size = static_cast<int64_t>(index.host_bfs_label_size[promote_label]);
  auto label_offset = static_cast<int64_t>(index.host_bfs_label_offset[promote_label]);
  auto dram_bytes = bfs_label_dram_requested_bytes<data_t>(label_size, index.dataset_dim);
  auto device_rows = make_device_storage_from_host(res, host_storage.get(), dram_bytes);
  vecflow_log("Rebalancing BFS tiers: promoting label ", promote_label, " from DRAM to HBM");
  (void)finalize_loaded_bfs_label(res, index, promote_label, label_offset, label_size, device_rows);
}

template <typename data_t>
void record_bfs_label_access(shared_resources::configured_raft_resources& res,
                             cuvs::neighbors::vecflow::index<data_t>& index,
                             uint32_t label)
{
  if (index.bfs_cache == nullptr) { return; }

  bool should_rebalance = false;
  {
    std::lock_guard<std::mutex> lock(index.bfs_cache->access_mutex);
    if (label >= index.bfs_cache->access_counts.size()) {
      index.bfs_cache->access_counts.resize(label + 1, 0);
    }
    auto next_access_event = index.bfs_cache->access_events + 1;
    auto should_decay = index.bfs_rebalance_interval_queries > 0 &&
                        (next_access_event % index.bfs_rebalance_interval_queries == 0);
    if (should_decay) {
      for (auto& access_count : index.bfs_cache->access_counts) {
        access_count >>= 1;
      }
    }
    index.bfs_cache->access_counts[label] += 1;
    index.bfs_cache->access_events = next_access_event;
    should_rebalance = should_decay;
  }

  if (should_rebalance) { rebalance_bfs_label_cache(res, index); }
}

template <typename data_t>
inline void schedule_future_bfs_prefetch(
  cuvs::neighbors::vecflow::index<data_t>& index,
  const std::vector<uint32_t>& label_order,
  std::size_t current_label_position,
  std::unique_ptr<async_host_copy_request<data_t>>& prefetched_dram_request,
  uint32_t* prefetched_label)
{
  if (prefetched_label == nullptr || prefetched_dram_request != nullptr ||
      index.bfs_prefetch_max_bytes == 0 || index.bfs_cache == nullptr) {
    return;
  }

  for (auto future_index = current_label_position + 1; future_index < label_order.size();
       ++future_index) {
    auto future_label = label_order[future_index];
    if (future_label >= index.host_bfs_label_size.size() ||
        future_label >= index.host_bfs_label_offset.size()) {
      throw std::runtime_error("BFS prefetch label is out of bounds: " +
                               std::to_string(future_label));
    }

    auto future_label_size = static_cast<int64_t>(index.host_bfs_label_size[future_label]);
    if (future_label_size <= 0) { continue; }
    auto future_requested_bytes =
      bfs_label_dram_requested_bytes<data_t>(future_label_size, index.dataset_dim);
    if (future_requested_bytes == 0 || future_requested_bytes > index.bfs_prefetch_max_bytes) {
      continue;
    }
    if (bfs_label_tier(index, future_label) != 1) { continue; }

    std::shared_ptr<data_t> host_storage;
    {
      std::lock_guard<std::mutex> lock(index.bfs_cache->dram_mutex);
      auto cache_it = index.bfs_cache->dram_entries.find(future_label);
      if (cache_it != index.bfs_cache->dram_entries.end()) {
        index.bfs_cache->dram_lru.splice(
          index.bfs_cache->dram_lru.begin(), index.bfs_cache->dram_lru, cache_it->second.lru_it);
        host_storage = cache_it->second.storage;
      }
    }
    if (host_storage == nullptr) { continue; }

    auto future_label_offset = static_cast<int64_t>(index.host_bfs_label_offset[future_label]);
    prefetched_dram_request = std::make_unique<async_host_copy_request<data_t>>(
      std::move(host_storage),
      future_requested_bytes,
      future_label,
      future_label_offset,
      future_label_size,
      "bfs");
    *prefetched_label = future_label;
    return;
  }
}

template <typename data_t>
inline auto resolve_prefetched_or_load_bfs_label_index(
  shared_resources::configured_raft_resources& res,
  cuvs::neighbors::vecflow::index<data_t>& index,
  uint32_t label,
  std::unique_ptr<async_host_copy_request<data_t>>& prefetched_dram_request,
  uint32_t* prefetched_label,
  reusable_device_buffer<data_t>* reusable_buffer = nullptr)
  -> std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>>
{
  if (prefetched_label != nullptr && prefetched_dram_request != nullptr &&
      *prefetched_label == label) {
    if (label >= index.host_bfs_label_size.size() || label >= index.host_bfs_label_offset.size()) {
      throw std::runtime_error("Prefetched BFS label is out of bounds: " + std::to_string(label));
    }
    auto label_size = static_cast<int64_t>(index.host_bfs_label_size[label]);
    auto label_offset = static_cast<int64_t>(index.host_bfs_label_offset[label]);
    auto device_rows = prefetched_dram_request->wait_and_release();
    prefetched_dram_request.reset();
    *prefetched_label = UINT32_MAX;
    {
      std::lock_guard<std::mutex> lock(index.bfs_cache->dram_mutex);
      index.bfs_cache->dram_hits += 1;
    }
    return finalize_loaded_bfs_label(
      res, index, label, label_offset, label_size, std::move(device_rows), false);
  }
  return get_or_load_bfs_label_index(res, index, label, reusable_buffer);
}

template <typename data_t>
void search_bfs_with_tiered_cache(shared_resources::configured_raft_resources& res,
                                  cuvs::neighbors::vecflow::index<data_t>& index,
                                  QueryInfo<data_t>& query_info,
                                  raft::device_matrix_view<int64_t, int64_t> bfs_neighbors,
                                  raft::device_matrix_view<float, int64_t> bfs_distances,
                                  int topk,
                                  const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  auto stream = raft::resource::get_cuda_stream(res);
  auto query_dim = query_info.bfs_queries.extent(1);
  auto num_bfs_queries = static_cast<int64_t>(query_info.bfs_query_map.size());
  if (num_bfs_queries == 0) { return; }

  std::vector<uint32_t> host_query_labels(static_cast<std::size_t>(num_bfs_queries));
  raft::copy(host_query_labels.data(),
             query_info.bfs_query_labels.data_handle(),
             num_bfs_queries,
             stream);
  raft::resource::sync_stream(res);

  std::unordered_map<uint32_t, std::vector<uint32_t>> query_positions_by_label;
  std::vector<uint32_t> label_order;
  label_order.reserve(host_query_labels.size());
  for (uint32_t i = 0; i < host_query_labels.size(); ++i) {
    auto label = host_query_labels[i];
    auto [it, inserted] = query_positions_by_label.try_emplace(label);
    if (inserted) { label_order.push_back(label); }
    it->second.push_back(i);
  }

  std::stable_sort(label_order.begin(), label_order.end(), [&](uint32_t lhs, uint32_t rhs) {
    auto lhs_rank = bfs_label_tier(index, lhs);
    auto rhs_rank = bfs_label_tier(index, rhs);
    if (lhs_rank != rhs_rank) { return lhs_rank < rhs_rank; }
    return query_positions_by_label.at(lhs).size() > query_positions_by_label.at(rhs).size();
  });

  auto run_bfs_batch = [&](const std::vector<uint32_t>& batch_labels,
                           const std::vector<std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>>>&
                             batch_indices) {
    if (batch_labels.empty()) { return; }

    auto batch_query_count = int64_t{0};
    for (auto label : batch_labels) {
      batch_query_count += static_cast<int64_t>(query_positions_by_label.at(label).size());
    }
    if (batch_query_count == 0) { return; }

    std::vector<uint32_t> host_positions;
    std::vector<uint32_t> host_batch_query_labels;
    std::vector<uint32_t> host_batch_label_sizes;
    host_positions.reserve(static_cast<std::size_t>(batch_query_count));
    host_batch_query_labels.reserve(static_cast<std::size_t>(batch_query_count));
    host_batch_label_sizes.reserve(batch_labels.size());

    for (std::size_t local_label = 0; local_label < batch_labels.size(); ++local_label) {
      auto label = batch_labels[local_label];
      if (label >= index.host_bfs_label_size.size()) {
        throw std::runtime_error("Tiered BFS query label is out of bounds: " + std::to_string(label));
      }
      auto label_size = static_cast<int64_t>(index.host_bfs_label_size[label]);
      if (label_size <= 0) {
        throw std::runtime_error("Tiered BFS search received a non-BFS query label " +
                                 std::to_string(label));
      }
      record_bfs_label_access(res, index, label);
      host_batch_label_sizes.push_back(static_cast<uint32_t>(label_size));
      for (auto query_pos : query_positions_by_label.at(label)) {
        host_positions.push_back(query_pos);
        host_batch_query_labels.push_back(static_cast<uint32_t>(local_label));
      }
    }

    auto scratch_positions = raft::make_device_vector<uint32_t, int64_t>(res, batch_query_count);
    auto scratch_queries = raft::make_device_matrix<data_t, int64_t>(res, batch_query_count, query_dim);
    auto scratch_query_labels =
      raft::make_device_vector<uint32_t, int64_t>(res, batch_query_count);
    auto scratch_label_size = raft::make_device_vector<uint32_t, int64_t>(
      res, static_cast<int64_t>(host_batch_label_sizes.size()));
    auto scratch_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, batch_query_count, topk);
    auto scratch_distances = raft::make_device_matrix<float, int64_t>(res, batch_query_count, topk);

    raft::update_device(
      scratch_positions.data_handle(), host_positions.data(), batch_query_count, stream);
    raft::update_device(scratch_query_labels.data_handle(),
                        host_batch_query_labels.data(),
                        batch_query_count,
                        stream);
    raft::update_device(scratch_label_size.data_handle(),
                        host_batch_label_sizes.data(),
                        static_cast<int64_t>(host_batch_label_sizes.size()),
                        stream);

    auto total_values = batch_query_count * query_dim;
    auto block_size = 256;
    if (total_values > 0) {
      auto grid_size = static_cast<int>((total_values + block_size - 1) / block_size);
      gather_rows_kernel<<<grid_size, block_size, 0, stream>>>(query_info.bfs_queries.data_handle(),
                                                               scratch_positions.data_handle(),
                                                               scratch_queries.data_handle(),
                                                               batch_query_count,
                                                               query_dim);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    auto batch_queries_view = raft::make_device_matrix_view<const data_t, int64_t, raft::row_major>(
      scratch_queries.data_handle(), batch_query_count, query_dim);
    auto batch_query_labels_view =
      raft::make_device_vector_view<uint32_t, int64_t>(scratch_query_labels.data_handle(),
                                                       batch_query_count);
    auto batch_label_size_view =
      raft::make_device_vector_view<uint32_t, int64_t>(scratch_label_size.data_handle(),
                                                       static_cast<int64_t>(host_batch_label_sizes.size()));
    auto batch_neighbors_view =
      raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
        scratch_neighbors.data_handle(), batch_query_count, topk);
    auto batch_distances_view =
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(
        scratch_distances.data_handle(), batch_query_count, topk);

    auto merged_index = cuvs::neighbors::ivf_flat::index<data_t, int64_t>(
      res,
      batch_indices.front()->metric(),
      static_cast<uint32_t>(batch_indices.size()),
      batch_indices.front()->adaptive_centers(),
      batch_indices.front()->conservative_memory_allocation(),
      batch_indices.front()->dim());
    for (std::size_t i = 0; i < batch_indices.size(); ++i) {
      merged_index.lists()[i] = batch_indices[i] != nullptr ? batch_indices[i]->lists()[0] : nullptr;
    }
    raft::update_device(merged_index.list_sizes().data_handle(),
                        host_batch_label_sizes.data(),
                        static_cast<uint32_t>(host_batch_label_sizes.size()),
                        stream);
    cuvs::neighbors::ivf_flat::helpers::recompute_internal_state(res, &merged_index);

    search_filtered_bfs(res,
                        merged_index,
                        batch_queries_view,
                        batch_query_labels_view,
                        batch_label_size_view,
                        batch_neighbors_view,
                        batch_distances_view,
                        cuvs::distance::DistanceType::L2Unexpanded,
                        sample_filter);

    auto total_results = batch_query_count * topk;
    if (total_results > 0) {
      auto result_grid = static_cast<int>((total_results + block_size - 1) / block_size);
      scatter_group_results_kernel<int64_t><<<result_grid, block_size, 0, stream>>>(
        bfs_neighbors.data_handle(),
        bfs_distances.data_handle(),
        scratch_neighbors.data_handle(),
        scratch_distances.data_handle(),
        scratch_positions.data_handle(),
        static_cast<int>(batch_query_count),
        topk);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  };

  std::vector<uint32_t> hbm_labels;
  std::vector<std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>>> hbm_indices;
  std::vector<uint32_t> non_hbm_labels;
  hbm_labels.reserve(label_order.size());
  hbm_indices.reserve(label_order.size());
  non_hbm_labels.reserve(label_order.size());

  for (auto label : label_order) {
    std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>> cached_index;
    if (index.bfs_cache != nullptr) {
      std::lock_guard<std::mutex> lock(index.bfs_cache->hbm_mutex);
      auto cache_it = index.bfs_cache->hbm_entries.find(label);
      if (cache_it != index.bfs_cache->hbm_entries.end()) {
        index.bfs_cache->hbm_lru.splice(
          index.bfs_cache->hbm_lru.begin(), index.bfs_cache->hbm_lru, cache_it->second.lru_it);
        index.bfs_cache->hbm_hits += 1;
        cached_index = cache_it->second.index;
      }
    }
    if (cached_index != nullptr) {
      hbm_labels.push_back(label);
      hbm_indices.push_back(std::move(cached_index));
    } else {
      non_hbm_labels.push_back(label);
    }
  }

  if (!hbm_labels.empty()) { run_bfs_batch(hbm_labels, hbm_indices); }

  // --- Phase 4.3: batch DRAM labels instead of serial per-label processing ---
  std::vector<uint32_t> dram_labels;
  std::vector<std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>>> dram_indices;
  std::vector<uint32_t> ssd_labels;
  dram_labels.reserve(non_hbm_labels.size());
  dram_indices.reserve(non_hbm_labels.size());
  ssd_labels.reserve(non_hbm_labels.size());

  reusable_device_buffer<data_t> dram_staging_buffer;
  for (auto label : non_hbm_labels) {
    if (label >= index.host_bfs_label_size.size() || label >= index.host_bfs_label_offset.size()) {
      throw std::runtime_error("Tiered BFS query label is out of bounds: " + std::to_string(label));
    }
    auto label_size = static_cast<int64_t>(index.host_bfs_label_size[label]);
    auto label_offset = static_cast<int64_t>(index.host_bfs_label_offset[label]);
    if (label_size <= 0) {
      throw std::runtime_error("Tiered BFS search received a non-BFS query label " +
                               std::to_string(label));
    }
    record_bfs_label_access(res, index, label);

    auto dram_bytes = bfs_label_dram_requested_bytes<data_t>(label_size, index.dataset_dim);
    std::shared_ptr<data_t> host_storage;
    if (index.bfs_cache != nullptr && index.bfs_dram_capacity_bytes > 0) {
      std::lock_guard<std::mutex> lock(index.bfs_cache->dram_mutex);
      auto cache_it = index.bfs_cache->dram_entries.find(label);
      if (cache_it != index.bfs_cache->dram_entries.end()) {
        index.bfs_cache->dram_lru.splice(
          index.bfs_cache->dram_lru.begin(), index.bfs_cache->dram_lru, cache_it->second.lru_it);
        host_storage = cache_it->second.storage;
        index.bfs_cache->dram_hits += 1;
      }
    }
    if (host_storage != nullptr) {
      auto device_rows = dram_staging_buffer.copy_from_host(res, host_storage.get(), dram_bytes);
      auto label_index = finalize_loaded_bfs_label(
        res, index, label, label_offset, label_size, std::move(device_rows), false);
      if (label_index != nullptr) {
        dram_labels.push_back(label);
        dram_indices.push_back(std::move(label_index));
      }
    } else {
      ssd_labels.push_back(label);
    }
  }

  if (!dram_labels.empty()) { run_bfs_batch(dram_labels, dram_indices); }

  // SSD labels still use serial path (rare under warm cache)
  for (std::size_t label_position = 0; label_position < ssd_labels.size(); ++label_position) {
    auto label = ssd_labels[label_position];
    auto label_size = static_cast<int64_t>(index.host_bfs_label_size[label]);

    auto const& host_positions = query_positions_by_label.at(label);
    auto group_size = static_cast<int64_t>(host_positions.size());
    auto scratch_positions = raft::make_device_vector<uint32_t, int64_t>(res, group_size);
    auto scratch_queries = raft::make_device_matrix<data_t, int64_t>(res, group_size, query_dim);
    auto scratch_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, group_size);
    auto scratch_label_size = raft::make_device_vector<uint32_t, int64_t>(res, 1);
    auto scratch_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, group_size, topk);
    auto scratch_distances = raft::make_device_matrix<float, int64_t>(res, group_size, topk);

    raft::update_device(
      scratch_positions.data_handle(), host_positions.data(), group_size, stream);

    auto total_values = group_size * query_dim;
    auto block_size = 256;
    if (total_values > 0) {
      auto grid_size = static_cast<int>((total_values + block_size - 1) / block_size);
      gather_rows_kernel<<<grid_size, block_size, 0, stream>>>(query_info.bfs_queries.data_handle(),
                                                               scratch_positions.data_handle(),
                                                               scratch_queries.data_handle(),
                                                               group_size,
                                                               query_dim);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    RAFT_CUDA_TRY(cudaMemsetAsync(
      scratch_query_labels.data_handle(), 0, group_size * sizeof(uint32_t), stream));
    uint32_t host_label_size_u32 = static_cast<uint32_t>(label_size);
    raft::update_device(scratch_label_size.data_handle(), &host_label_size_u32, 1, stream);

    if (index.bfs_cache != nullptr) {
      std::lock_guard<std::mutex> lock(index.bfs_cache->access_mutex);
      index.bfs_cache->ssd_loads += 1;
    }
    auto label_offset = static_cast<int64_t>(index.host_bfs_label_offset[label]);
    auto device_rows = load_ibin_rows_to_device<data_t>(
      res, index.bfs_dataset_cache_fname, index.bfs_total_rows, index.dataset_dim, label_offset, label_size);
    auto label_index = finalize_loaded_bfs_label(
      res, index, label, label_offset, label_size, std::move(device_rows));

    auto group_queries_view = raft::make_device_matrix_view<const data_t, int64_t, raft::row_major>(
      scratch_queries.data_handle(), group_size, query_dim);
    auto group_query_labels_view =
      raft::make_device_vector_view<uint32_t, int64_t>(scratch_query_labels.data_handle(), group_size);
    auto group_label_size_view =
      raft::make_device_vector_view<uint32_t, int64_t>(scratch_label_size.data_handle(), 1);
    auto group_neighbors_view =
      raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
        scratch_neighbors.data_handle(), group_size, topk);
    auto group_distances_view =
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(
        scratch_distances.data_handle(), group_size, topk);

    search_filtered_bfs(res,
                        *label_index,
                        group_queries_view,
                        group_query_labels_view,
                        group_label_size_view,
                        group_neighbors_view,
                        group_distances_view,
                        cuvs::distance::DistanceType::L2Unexpanded,
                        sample_filter);

    auto total_results = group_size * topk;
    auto result_grid = static_cast<int>((total_results + block_size - 1) / block_size);
    scatter_group_results_kernel<int64_t><<<result_grid, block_size, 0, stream>>>(
      bfs_neighbors.data_handle(),
      bfs_distances.data_handle(),
      scratch_neighbors.data_handle(),
      scratch_distances.data_handle(),
      scratch_positions.data_handle(),
      static_cast<int>(group_size),
      topk);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

template <typename data_t>
void rebalance_phoenix_label_cache(shared_resources::configured_raft_resources& res,
                                   cuvs::neighbors::vecflow::index<data_t>& index)
{
  if (index.phoenix_label_cache == nullptr || index.phoenix_label_cache_capacity_bytes == 0 ||
      index.phoenix_label_dram_cache_capacity_bytes == 0) {
    return;
  }

  std::shared_ptr<uint32_t> host_graph_storage;
  uint32_t promote_label = std::numeric_limits<uint32_t>::max();
  int64_t promote_label_offset = 0;
  int64_t promote_label_size = 0;
  std::size_t promote_bytes = 0;
  double best_dram_score = 0.0;
  double worst_hbm_score = std::numeric_limits<double>::infinity();
  bool has_hbm_entries = false;

  {
    std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
    for (auto const& [label, entry] : index.phoenix_label_cache->graphs) {
      has_hbm_entries = true;
      worst_hbm_score = std::min(
        worst_hbm_score, phoenix_label_cache_score_locked(index, label, entry.bytes));
    }

    for (auto const& [label, entry] : index.phoenix_label_cache->host_graphs) {
      auto score = phoenix_label_cache_score_locked(index, label, entry.bytes);
      if (score > best_dram_score) {
        best_dram_score = score;
        promote_label = label;
        host_graph_storage = entry.storage;
      }
    }
  }

  if (promote_label == std::numeric_limits<uint32_t>::max() || host_graph_storage == nullptr ||
      (has_hbm_entries && best_dram_score <= worst_hbm_score) ||
      promote_label >= index.host_cagra_label_size.size()) {
    return;
  }

  promote_label_size = static_cast<int64_t>(index.host_cagra_label_size[promote_label]);
  promote_label_offset = static_cast<int64_t>(index.host_cagra_label_offset[promote_label]);
  promote_bytes = label_graph_requested_bytes(index, promote_label_size);
  if (promote_bytes == 0) { return; }

  auto device_graph_storage =
    make_device_storage_from_host(res, host_graph_storage.get(), promote_bytes);
  vecflow_log("Rebalancing Phoenix tiers: promoting label ", promote_label, " from DRAM to HBM");
  cache_label_graph_in_hbm(
    res,
    index,
    promote_label,
    promote_label_offset,
    promote_label_size,
    promote_bytes,
    device_graph_storage);
}

template <typename data_t>
void rebalance_phoenix_label_dataset_cache(shared_resources::configured_raft_resources& res,
                                           cuvs::neighbors::vecflow::index<data_t>& index,
                                           int64_t dim)
{
  if (index.phoenix_label_dataset_cache == nullptr ||
      index.phoenix_label_dataset_cache_capacity_bytes == 0 ||
      index.phoenix_label_dataset_dram_cache_capacity_bytes == 0) {
    return;
  }

  std::shared_ptr<data_t> host_dataset_storage;
  uint32_t promote_label = std::numeric_limits<uint32_t>::max();
  int64_t promote_label_offset = 0;
  int64_t promote_label_size = 0;
  std::size_t promote_bytes = 0;
  double best_dram_score = 0.0;
  double worst_hbm_score = std::numeric_limits<double>::infinity();
  bool has_hbm_entries = false;

  {
    std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
    for (auto const& [label, entry] : index.phoenix_label_dataset_cache->datasets) {
      has_hbm_entries = true;
      worst_hbm_score = std::min(
        worst_hbm_score, phoenix_label_cache_score_locked(index, label, entry.bytes));
    }

    for (auto const& [label, entry] : index.phoenix_label_dataset_cache->host_datasets) {
      auto score = phoenix_label_cache_score_locked(index, label, entry.bytes);
      if (score > best_dram_score) {
        best_dram_score = score;
        promote_label = label;
        host_dataset_storage = entry.storage;
      }
    }
  }

  if (promote_label == std::numeric_limits<uint32_t>::max() || host_dataset_storage == nullptr ||
      (has_hbm_entries && best_dram_score <= worst_hbm_score) ||
      promote_label >= index.host_cagra_label_size.size()) {
    return;
  }

  promote_label_size = static_cast<int64_t>(index.host_cagra_label_size[promote_label]);
  promote_label_offset = static_cast<int64_t>(index.host_cagra_label_offset[promote_label]);
  promote_bytes = label_dataset_requested_bytes<data_t>(promote_label_size, dim);
  if (promote_bytes == 0) { return; }

  auto device_dataset_storage =
    make_device_storage_from_host(res, host_dataset_storage.get(), promote_bytes);
  vecflow_log(
    "Rebalancing Phoenix dataset tiers: promoting label ", promote_label, " from DRAM to HBM");
  cache_label_dataset_in_hbm(res,
                             index,
                             promote_label,
                             promote_label_offset,
                             promote_label_size,
                             promote_bytes,
                             device_dataset_storage);
}

template <typename data_t>
void record_phoenix_label_access(shared_resources::configured_raft_resources& res,
                                 cuvs::neighbors::vecflow::index<data_t>& index,
                                 uint32_t label,
                                 std::optional<int64_t> dataset_dim = std::nullopt)
{
  if (index.phoenix_label_cache == nullptr) { return; }

  bool should_rebalance = false;
  {
    std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
    if (label >= index.phoenix_label_cache->access_counts.size()) {
      index.phoenix_label_cache->access_counts.resize(label + 1, 0);
    }
    index.phoenix_label_cache->access_counts[label] += 1;
    index.phoenix_label_cache->access_events += 1;
    should_rebalance = index.phoenix_rebalance_interval_queries > 0 &&
                       (index.phoenix_label_cache->access_events %
                          index.phoenix_rebalance_interval_queries) == 0;
  }

  if (should_rebalance) {
    rebalance_phoenix_label_cache(res, index);
    if (dataset_dim.has_value()) {
      rebalance_phoenix_label_dataset_cache(res, index, *dataset_dim);
    }
  }
}

template <typename data_t>
auto get_or_load_phoenix_label_graph(shared_resources::configured_raft_resources& res,
                                     cuvs::neighbors::vecflow::index<data_t>& index,
                                     uint32_t label,
                                     int64_t total_graph_rows,
                                     int64_t label_offset,
                                     int64_t label_size) -> std::shared_ptr<uint32_t>
{
#ifndef CUVS_VECFLOW_PHOENIX_ENABLED
  throw std::runtime_error(
    "Phoenix label-load path requested, but VecFlow was built without Phoenix support.");
#else
  auto requested_bytes = label_graph_requested_bytes(index, label_size);

  if (index.phoenix_label_cache != nullptr && index.phoenix_label_cache_capacity_bytes > 0) {
    {
      std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
      auto cache_it = index.phoenix_label_cache->graphs.find(label);
      if (cache_it != index.phoenix_label_cache->graphs.end()) {
        index.phoenix_label_cache->lru_labels.splice(index.phoenix_label_cache->lru_labels.begin(),
                                                     index.phoenix_label_cache->lru_labels,
                                                     cache_it->second.lru_it);
        vecflow_log("Reusing cached graph rows [",
                    label_offset,
                    ", ",
                    (label_offset + label_size),
                    ") for label ",
                    label);
        index.phoenix_label_cache->hbm_hits += 1;
        return cache_it->second.storage;
      }
    }
  }

  if (index.phoenix_label_cache != nullptr && index.phoenix_label_dram_cache_capacity_bytes > 0) {
    std::shared_ptr<uint32_t> host_graph_storage;
    {
      std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
      auto cache_it = index.phoenix_label_cache->host_graphs.find(label);
      if (cache_it != index.phoenix_label_cache->host_graphs.end()) {
        index.phoenix_label_cache->host_lru_labels.splice(
          index.phoenix_label_cache->host_lru_labels.begin(),
          index.phoenix_label_cache->host_lru_labels,
          cache_it->second.lru_it);
        host_graph_storage = cache_it->second.storage;
      }
    }
    if (host_graph_storage != nullptr) {
      vecflow_log("Promoting cached host graph rows [",
                  label_offset,
                  ", ",
                  (label_offset + label_size),
                  ") for label ",
                  label);
      auto device_graph_storage =
        make_device_storage_from_host(res, host_graph_storage.get(), requested_bytes);
      index.phoenix_label_cache->dram_hits += 1;
      cache_label_graph_in_hbm(
        res, index, label, label_offset, label_size, requested_bytes, device_graph_storage);
      return device_graph_storage;
    }
  }

  index.phoenix_label_cache->ssd_loads += 1;
  auto graph_storage = detail::phoenix::load_ibin_graph_rows_to_device(res,
                                                                       index.cagra_graph_cache_fname,
                                                                       total_graph_rows,
                                                                       cached_graph_storage_width(index),
                                                                       label_offset,
                                                                       label_size);
  return finalize_loaded_phoenix_label_graph(
    res, index, label, label_offset, label_size, std::move(graph_storage));
#endif
}

template <typename data_t>
auto get_or_load_phoenix_label_dataset(shared_resources::configured_raft_resources& res,
                                       cuvs::neighbors::vecflow::index<data_t>& index,
                                       uint32_t label,
                                       int64_t total_dataset_rows,
                                       int64_t label_offset,
                                       int64_t label_size,
                                       int64_t dim) -> std::shared_ptr<data_t>
{
#ifndef CUVS_VECFLOW_PHOENIX_ENABLED
  throw std::runtime_error(
    "Phoenix label-load path requested, but VecFlow was built without Phoenix support.");
#else
  if (index.cagra_dataset_cache_fname.empty()) {
    throw std::runtime_error("Phoenix dataset label-load path requires a cached dataset file.");
  }

  auto requested_bytes = label_dataset_requested_bytes<data_t>(label_size, dim);

  if (index.phoenix_label_dataset_cache != nullptr &&
      index.phoenix_label_dataset_cache_capacity_bytes > 0) {
    {
      std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
      auto cache_it = index.phoenix_label_dataset_cache->datasets.find(label);
      if (cache_it != index.phoenix_label_dataset_cache->datasets.end()) {
        index.phoenix_label_dataset_cache->lru_labels.splice(
          index.phoenix_label_dataset_cache->lru_labels.begin(),
          index.phoenix_label_dataset_cache->lru_labels,
          cache_it->second.lru_it);
        vecflow_log("Reusing cached dataset rows [",
                    label_offset,
                    ", ",
                    (label_offset + label_size),
                    ") for label ",
                    label);
        index.phoenix_label_dataset_cache->hbm_hits += 1;
        return cache_it->second.storage;
      }
    }
  }

  if (index.phoenix_label_dataset_cache != nullptr &&
      index.phoenix_label_dataset_dram_cache_capacity_bytes > 0) {
    std::shared_ptr<data_t> host_dataset_storage;
    {
      std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
      auto cache_it = index.phoenix_label_dataset_cache->host_datasets.find(label);
      if (cache_it != index.phoenix_label_dataset_cache->host_datasets.end()) {
        index.phoenix_label_dataset_cache->host_lru_labels.splice(
          index.phoenix_label_dataset_cache->host_lru_labels.begin(),
          index.phoenix_label_dataset_cache->host_lru_labels,
          cache_it->second.lru_it);
        host_dataset_storage = cache_it->second.storage;
      }
    }
    if (host_dataset_storage != nullptr) {
      vecflow_log("Promoting cached host dataset rows [",
                  label_offset,
                  ", ",
                  (label_offset + label_size),
                  ") for label ",
                  label);
      auto device_dataset_storage =
        make_device_storage_from_host(res, host_dataset_storage.get(), requested_bytes);
      index.phoenix_label_dataset_cache->dram_hits += 1;
      cache_label_dataset_in_hbm(
        res, index, label, label_offset, label_size, requested_bytes, device_dataset_storage);
      return device_dataset_storage;
    }
  }

  index.phoenix_label_dataset_cache->ssd_loads += 1;
  auto dataset_storage = detail::phoenix::load_ibin_rows_to_device<data_t>(res,
                                                                            index.cagra_dataset_cache_fname,
                                                                            total_dataset_rows,
                                                                            dim,
                                                                            label_offset,
                                                                            label_size);
  return finalize_loaded_phoenix_label_dataset(
    res, index, label, label_offset, label_size, dim, std::move(dataset_storage));
#endif
}

#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
template <typename data_t>
inline void schedule_future_dataset_prefetch(
  cuvs::neighbors::vecflow::index<data_t>& index,
  const std::vector<uint32_t>& label_order,
  std::size_t current_label_position,
  int64_t total_dataset_rows,
  int64_t dim,
  std::unique_ptr<async_host_copy_request<data_t>>& prefetched_dram_request,
  std::unique_ptr<detail::phoenix::async_ibin_rows_request<data_t>>& prefetched_ssd_request,
  uint32_t* prefetched_label)
{
  if (prefetched_label == nullptr || prefetched_dram_request != nullptr ||
      prefetched_ssd_request != nullptr) {
    return;
  }

  auto prefetch_budget_bytes = detail::phoenix::phoenix_label_dataset_prefetch_max_bytes();
  if (prefetch_budget_bytes == 0 || index.cagra_dataset_cache_fname.empty()) { return; }

  for (auto future_index = current_label_position + 1; future_index < label_order.size();
       ++future_index) {
    auto future_label = label_order[future_index];
    if (future_label >= index.host_cagra_label_size.size()) {
      throw std::runtime_error("Phoenix dataset prefetch label is out of bounds: " +
                               std::to_string(future_label));
    }

    auto future_label_size = static_cast<int64_t>(index.host_cagra_label_size[future_label]);
    if (future_label_size <= 0) { continue; }
    auto future_requested_bytes =
      label_dataset_requested_bytes<data_t>(future_label_size, dim);
    if (future_requested_bytes == 0 || future_requested_bytes > prefetch_budget_bytes) {
      continue;
    }

    auto future_label_offset = static_cast<int64_t>(index.host_cagra_label_offset[future_label]);
    auto future_tier = phoenix_label_dataset_tier(index, future_label);
    if (future_tier == 1) {
      std::shared_ptr<data_t> host_dataset_storage;
      {
        std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
        auto cache_it = index.phoenix_label_dataset_cache->host_datasets.find(future_label);
        if (cache_it != index.phoenix_label_dataset_cache->host_datasets.end()) {
          index.phoenix_label_dataset_cache->host_lru_labels.splice(
            index.phoenix_label_dataset_cache->host_lru_labels.begin(),
            index.phoenix_label_dataset_cache->host_lru_labels,
            cache_it->second.lru_it);
          host_dataset_storage = cache_it->second.storage;
        }
      }
      if (host_dataset_storage == nullptr) { continue; }
      prefetched_dram_request = std::make_unique<async_host_copy_request<data_t>>(
        std::move(host_dataset_storage),
        future_requested_bytes,
        future_label,
        future_label_offset,
        future_label_size,
        "dataset");
      *prefetched_label = future_label;
      return;
    }

    if (future_tier == 2) {
      prefetched_ssd_request =
        std::make_unique<detail::phoenix::async_ibin_rows_request<data_t>>(
          index.cagra_dataset_cache_fname,
          total_dataset_rows,
          dim,
          future_label_offset,
          future_label_size);
      *prefetched_label = future_label;
      return;
    }
  }
}

template <typename data_t>
inline auto resolve_prefetched_or_load_phoenix_label_dataset(
  shared_resources::configured_raft_resources& res,
  cuvs::neighbors::vecflow::index<data_t>& index,
  uint32_t label,
  int64_t total_dataset_rows,
  int64_t label_offset,
  int64_t label_size,
  int64_t dim,
  std::unique_ptr<async_host_copy_request<data_t>>& prefetched_dram_request,
  std::unique_ptr<detail::phoenix::async_ibin_rows_request<data_t>>& prefetched_ssd_request,
  uint32_t* prefetched_label) -> std::shared_ptr<data_t>
{
  if (prefetched_label != nullptr && prefetched_dram_request != nullptr &&
      *prefetched_label == label) {
    vecflow_log("Using prefetched cached host dataset rows [",
                label_offset,
                ", ",
                (label_offset + label_size),
                ") for label ",
                label);
    auto dataset_storage = prefetched_dram_request->wait_and_release();
    prefetched_dram_request.reset();
    *prefetched_label = UINT32_MAX;
    index.phoenix_label_dataset_cache->dram_hits += 1;
    return finalize_loaded_phoenix_label_dataset(
      res, index, label, label_offset, label_size, dim, std::move(dataset_storage));
  }

  if (prefetched_label != nullptr && prefetched_ssd_request != nullptr &&
      *prefetched_label == label) {
    vecflow_log("Using prefetched dataset rows [",
                label_offset,
                ", ",
                (label_offset + label_size),
                ") for label ",
                label);
    auto dataset_storage = prefetched_ssd_request->wait_and_release();
    prefetched_ssd_request.reset();
    *prefetched_label = UINT32_MAX;
    index.phoenix_label_dataset_cache->ssd_loads += 1;
    return finalize_loaded_phoenix_label_dataset(
      res, index, label, label_offset, label_size, dim, std::move(dataset_storage));
  }

  return get_or_load_phoenix_label_dataset(
    res, index, label, total_dataset_rows, label_offset, label_size, dim);
}
#endif

template <typename data_t>
void search_cagra_with_phoenix_label_load(
  shared_resources::configured_raft_resources& res,
  cuvs::neighbors::vecflow::index<data_t>& index,
  const cuvs::neighbors::cagra::search_params& search_params,
  QueryInfo<data_t>& query_info,
  raft::device_matrix_view<uint32_t, int64_t> cagra_neighbors,
  raft::device_matrix_view<float, int64_t> cagra_distances,
  int topk,
  const cuvs::neighbors::filtering::base_filter& sample_filter)
{
#ifndef CUVS_VECFLOW_PHOENIX_ENABLED
  throw std::runtime_error(
    "Phoenix label-load path requested, but VecFlow was built without Phoenix support.");
#else
  if (index.cagra_graph_cache_fname.empty()) {
    throw std::runtime_error("Phoenix label-load path requires a cached IVF-Graph file.");
  }
  if (!std::filesystem::exists(index.cagra_graph_cache_fname)) {
    throw std::runtime_error("Phoenix label-load graph file does not exist: " +
                             index.cagra_graph_cache_fname);
  }
  if (index.cagra_graph_degree <= 0) {
    throw std::runtime_error("Phoenix label-load path requires a positive cached graph degree.");
  }
  if (index.host_cagra_label_size.empty() || index.host_cagra_label_offset.empty()) {
    throw std::runtime_error("Phoenix label-load path requires host-side per-label graph metadata.");
  }
  auto use_dataset_label_load =
    !index.cagra_dataset_cache_fname.empty() &&
    std::filesystem::exists(index.cagra_dataset_cache_fname);
  if (!use_dataset_label_load && index.ivf_graph_index.dataset().extent(0) == 0) {
    throw std::runtime_error(
      "Phoenix label-load path requires either the dataset to stay attached or a packed dataset cache.");
  }

  auto stream = raft::resource::get_cuda_stream(res);
  auto query_dim = query_info.cagra_queries.extent(1);
  auto total_graph_rows = static_cast<int64_t>(index.cagra_index_map.size());
  auto num_cagra_queries = static_cast<int64_t>(query_info.cagra_query_map.size());
  if (num_cagra_queries == 0) { return; }

  std::vector<uint32_t> host_query_labels(static_cast<std::size_t>(num_cagra_queries));
  raft::copy(host_query_labels.data(),
             query_info.cagra_query_labels.data_handle(),
             num_cagra_queries,
             stream);
  raft::resource::sync_stream(res);

  std::unordered_map<uint32_t, std::vector<uint32_t>> query_positions_by_label;
  std::vector<uint32_t> label_order;
  label_order.reserve(host_query_labels.size());
  for (uint32_t i = 0; i < host_query_labels.size(); ++i) {
    auto label = host_query_labels[i];
    auto [it, inserted] = query_positions_by_label.try_emplace(label);
    if (inserted) { label_order.push_back(label); }
    it->second.push_back(i);
  }

  if (index.phoenix_label_cache != nullptr) {
    std::stable_sort(label_order.begin(), label_order.end(), [&](uint32_t lhs, uint32_t rhs) {
      auto lhs_rank = phoenix_label_graph_tier(index, lhs);
      auto rhs_rank = phoenix_label_graph_tier(index, rhs);
      if (lhs_rank != rhs_rank) { return lhs_rank < rhs_rank; }
      return query_positions_by_label.at(lhs).size() > query_positions_by_label.at(rhs).size();
    });
  }

  auto try_get_hbm_graph = [&](uint32_t label) -> std::shared_ptr<uint32_t> {
    if (index.phoenix_label_cache == nullptr) { return {}; }
    std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
    auto cache_it = index.phoenix_label_cache->graphs.find(label);
    if (cache_it == index.phoenix_label_cache->graphs.end()) { return {}; }
    index.phoenix_label_cache->lru_labels.splice(index.phoenix_label_cache->lru_labels.begin(),
                                                 index.phoenix_label_cache->lru_labels,
                                                 cache_it->second.lru_it);
    index.phoenix_label_cache->hbm_hits += 1;
    return cache_it->second.storage;
  };

  auto try_get_hbm_dataset = [&](uint32_t label) -> std::shared_ptr<data_t> {
    if (!use_dataset_label_load || index.phoenix_label_dataset_cache == nullptr) { return {}; }
    std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
    auto cache_it = index.phoenix_label_dataset_cache->datasets.find(label);
    if (cache_it == index.phoenix_label_dataset_cache->datasets.end()) { return {}; }
    index.phoenix_label_dataset_cache->lru_labels.splice(
      index.phoenix_label_dataset_cache->lru_labels.begin(),
      index.phoenix_label_dataset_cache->lru_labels,
      cache_it->second.lru_it);
    index.phoenix_label_dataset_cache->hbm_hits += 1;
    return cache_it->second.storage;
  };

  auto should_run_hbm_batch = [&](const std::vector<uint32_t>& batch_labels) {
    constexpr std::size_t kPhoenixHbmBatchMinLabels = 3;
    constexpr std::size_t kPhoenixHbmBatchMaxD2DBytes = std::size_t{512} << 20;
    if (batch_labels.size() < kPhoenixHbmBatchMinLabels) { return false; }

    auto batch_total_rows = int64_t{0};
    for (auto label : batch_labels) {
      batch_total_rows += static_cast<int64_t>(index.host_cagra_label_size[label]);
    }
    if (batch_total_rows <= 0) { return false; }

    auto merged_bytes = label_graph_requested_bytes(index, batch_total_rows);
    if (use_dataset_label_load) {
      merged_bytes += label_dataset_requested_bytes<data_t>(batch_total_rows, query_dim);
    }
    return merged_bytes <= kPhoenixHbmBatchMaxD2DBytes;
  };

  auto run_hbm_batch = [&](const std::vector<uint32_t>& batch_labels,
                           const std::vector<std::shared_ptr<uint32_t>>& batch_graphs,
                           const std::vector<std::shared_ptr<data_t>>& batch_datasets) {
    if (batch_labels.empty()) { return; }

    auto batch_query_count = int64_t{0};
    auto batch_total_rows = int64_t{0};
    std::vector<uint32_t> host_positions;
    std::vector<uint32_t> host_query_label_ids;
    std::vector<uint32_t> host_label_sizes;
    std::vector<uint32_t> host_label_offsets;
    host_label_sizes.reserve(batch_labels.size());
    host_label_offsets.reserve(batch_labels.size());

    for (std::size_t local_label = 0; local_label < batch_labels.size(); ++local_label) {
      auto label = batch_labels[local_label];
      auto label_size = static_cast<int64_t>(index.host_cagra_label_size[label]);
      if (label_size <= 0) {
        throw std::runtime_error("Phoenix label-load received a CAGRA query for non-CAGRA label " +
                                 std::to_string(label));
      }
      record_phoenix_label_access(res, index, label, query_dim);
      host_label_offsets.push_back(static_cast<uint32_t>(batch_total_rows));
      host_label_sizes.push_back(static_cast<uint32_t>(label_size));
      batch_total_rows += label_size;
      auto const& positions = query_positions_by_label.at(label);
      batch_query_count += static_cast<int64_t>(positions.size());
      for (auto query_pos : positions) {
        host_positions.push_back(query_pos);
        host_query_label_ids.push_back(static_cast<uint32_t>(local_label));
      }
    }
    if (batch_query_count == 0 || batch_total_rows == 0) { return; }

    auto scratch_positions = raft::make_device_vector<uint32_t, int64_t>(res, batch_query_count);
    auto scratch_queries = raft::make_device_matrix<data_t, int64_t>(res, batch_query_count, query_dim);
    auto scratch_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, batch_query_count);
    auto scratch_index_map = raft::make_device_vector<uint32_t, int64_t>(res, batch_total_rows);
    auto scratch_label_size = raft::make_device_vector<uint32_t, int64_t>(
      res, static_cast<int64_t>(host_label_sizes.size()));
    auto scratch_label_offset = raft::make_device_vector<uint32_t, int64_t>(
      res, static_cast<int64_t>(host_label_offsets.size()));
    auto scratch_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, batch_query_count, topk);
    auto scratch_distances = raft::make_device_matrix<float, int64_t>(res, batch_query_count, topk);
    auto merged_graph = raft::make_device_matrix<uint32_t, int64_t>(
      res, batch_total_rows, cached_graph_storage_width(index));

    raft::update_device(
      scratch_positions.data_handle(), host_positions.data(), batch_query_count, stream);
    raft::update_device(scratch_query_labels.data_handle(),
                        host_query_label_ids.data(),
                        batch_query_count,
                        stream);
    raft::update_device(scratch_label_size.data_handle(),
                        host_label_sizes.data(),
                        static_cast<int64_t>(host_label_sizes.size()),
                        stream);
    raft::update_device(scratch_label_offset.data_handle(),
                        host_label_offsets.data(),
                        static_cast<int64_t>(host_label_offsets.size()),
                        stream);

    auto total_values = batch_query_count * query_dim;
    auto block_size = 256;
    if (total_values > 0) {
      auto grid_size = static_cast<int>((total_values + block_size - 1) / block_size);
      gather_rows_kernel<<<grid_size, block_size, 0, stream>>>(query_info.cagra_queries.data_handle(),
                                                               scratch_positions.data_handle(),
                                                               scratch_queries.data_handle(),
                                                               batch_query_count,
                                                               query_dim);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    std::optional<raft::device_matrix<data_t, int64_t>> merged_dataset;
    if (use_dataset_label_load) {
      merged_dataset.emplace(
        raft::make_device_matrix<data_t, int64_t>(res, batch_total_rows, query_dim));
    }

    auto running_offset = int64_t{0};
    for (std::size_t local_label = 0; local_label < batch_labels.size(); ++local_label) {
      auto label = batch_labels[local_label];
      auto label_size = static_cast<int64_t>(index.host_cagra_label_size[label]);
      auto label_offset = static_cast<int64_t>(index.host_cagra_label_offset[label]);
      raft::copy(merged_graph.data_handle() +
                   running_offset * static_cast<int64_t>(cached_graph_storage_width(index)),
                 batch_graphs[local_label].get(),
                 label_size * static_cast<int64_t>(cached_graph_storage_width(index)),
                 stream);
      raft::copy(scratch_index_map.data_handle() + running_offset,
                 index.cagra_index_map.data_handle() + label_offset,
                 label_size,
                 stream);
      if (use_dataset_label_load) {
        raft::copy(merged_dataset->data_handle() + running_offset * query_dim,
                   batch_datasets[local_label].get(),
                   label_size * query_dim,
                   stream);
      }
      running_offset += label_size;
    }

    auto batch_queries_view = raft::make_device_matrix_view<const data_t, int64_t, raft::row_major>(
      scratch_queries.data_handle(), batch_query_count, query_dim);
    auto batch_query_labels_view = raft::make_device_vector_view<uint32_t, int64_t>(
      scratch_query_labels.data_handle(), batch_query_count);
    auto batch_index_map_view = raft::make_device_vector_view<uint32_t, int64_t>(
      scratch_index_map.data_handle(), batch_total_rows);
    auto batch_label_size_view = raft::make_device_vector_view<uint32_t, int64_t>(
      scratch_label_size.data_handle(), static_cast<int64_t>(host_label_sizes.size()));
    auto batch_label_offset_view = raft::make_device_vector_view<uint32_t, int64_t>(
      scratch_label_offset.data_handle(), static_cast<int64_t>(host_label_offsets.size()));
    auto batch_neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
      scratch_neighbors.data_handle(), batch_query_count, topk);
    auto batch_distances_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
      scratch_distances.data_handle(), batch_query_count, topk);

    auto batched_index =
      cuvs::neighbors::cagra::index<data_t, uint32_t>(res, index.ivf_graph_index.metric());
    if (use_dataset_label_load) {
      auto merged_dataset_view = raft::make_device_matrix_view<const data_t, int64_t, raft::row_major>(
        merged_dataset->data_handle(), batch_total_rows, query_dim);
      batched_index.update_dataset(res, merged_dataset_view);
    } else {
      batched_index.update_dataset(res, index.ivf_graph_index.dataset());
    }
    auto merged_graph_view = raft::make_device_matrix_view<const uint32_t, int64_t, raft::row_major>(
      merged_graph.data_handle(), batch_total_rows, cached_graph_storage_width(index));
    batched_index.update_graph(res, merged_graph_view);

    cagra::filtered_search(res,
                           search_params,
                           batched_index,
                           batch_queries_view,
                           batch_neighbors_view,
                           batch_distances_view,
                           batch_query_labels_view,
                           batch_index_map_view,
                           batch_label_size_view,
                           batch_label_offset_view,
                           sample_filter);

    auto total_results = batch_query_count * topk;
    if (total_results > 0) {
      auto result_grid = static_cast<int>((total_results + block_size - 1) / block_size);
      merge_neighbors_kernel<data_t, uint32_t><<<result_grid, block_size, 0, stream>>>(
        cagra_neighbors.data_handle(),
        cagra_distances.data_handle(),
        scratch_neighbors.data_handle(),
        scratch_distances.data_handle(),
        scratch_positions.data_handle(),
        batch_query_count,
        topk);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  };

  std::vector<uint32_t> hbm_labels;
  std::vector<uint32_t> non_hbm_labels;
  for (auto label : label_order) {
    auto graph_hbm = phoenix_label_graph_tier(index, label) == 0;
    auto dataset_hbm = !use_dataset_label_load || phoenix_label_dataset_tier(index, label) == 0;
    if (graph_hbm && dataset_hbm) {
      hbm_labels.push_back(label);
    } else {
      non_hbm_labels.push_back(label);
    }
  }

  auto processed_hbm_in_batch = false;
  if (!hbm_labels.empty() && should_run_hbm_batch(hbm_labels)) {
    std::vector<std::shared_ptr<uint32_t>> hbm_graphs;
    std::vector<std::shared_ptr<data_t>> hbm_datasets;
    hbm_graphs.reserve(hbm_labels.size());
    if (use_dataset_label_load) { hbm_datasets.reserve(hbm_labels.size()); }

    auto batch_ready = true;
    for (auto label : hbm_labels) {
      auto graph_storage = try_get_hbm_graph(label);
      auto dataset_storage = std::shared_ptr<data_t>{};
      if (use_dataset_label_load) { dataset_storage = try_get_hbm_dataset(label); }
      if (graph_storage == nullptr || (use_dataset_label_load && dataset_storage == nullptr)) {
        batch_ready = false;
        break;
      }
      hbm_graphs.push_back(std::move(graph_storage));
      if (use_dataset_label_load) { hbm_datasets.push_back(std::move(dataset_storage)); }
    }

    if (batch_ready) {
      run_hbm_batch(hbm_labels, hbm_graphs, hbm_datasets);
      processed_hbm_in_batch = true;
    }
  }

  auto const& serial_labels = processed_hbm_in_batch ? non_hbm_labels : label_order;

  auto prefetch_budget_bytes = detail::phoenix::phoenix_label_prefetch_max_bytes();
  std::unique_ptr<detail::phoenix::async_ibin_graph_rows_request> prefetched_ssd_request;
  std::unique_ptr<async_host_graph_copy_request> prefetched_dram_request;
  std::unique_ptr<detail::phoenix::async_ibin_rows_request<data_t>> prefetched_dataset_ssd_request;
  std::unique_ptr<async_host_copy_request<data_t>> prefetched_dataset_dram_request;
  uint32_t prefetched_label = UINT32_MAX;
  uint32_t prefetched_dataset_label = UINT32_MAX;
  auto has_prefetched_request = [&]() {
    return prefetched_ssd_request != nullptr || prefetched_dram_request != nullptr;
  };

  auto schedule_future_prefetch = [&](std::size_t current_label_position) {
    if (has_prefetched_request() || prefetch_budget_bytes == 0) { return; }

    for (auto future_index = current_label_position + 1; future_index < serial_labels.size();
         ++future_index) {
      auto future_label = serial_labels[future_index];
      if (future_label >= index.host_cagra_label_size.size()) {
        throw std::runtime_error("Phoenix label-load query label is out of bounds: " +
                                 std::to_string(future_label));
      }

      auto future_label_size = static_cast<int64_t>(index.host_cagra_label_size[future_label]);
      if (future_label_size <= 0) { continue; }
      auto future_requested_bytes = label_graph_requested_bytes(index, future_label_size);
      if (future_requested_bytes == 0 || future_requested_bytes > prefetch_budget_bytes) {
        continue;
      }
      auto future_label_offset = static_cast<int64_t>(index.host_cagra_label_offset[future_label]);
      auto future_tier = phoenix_label_graph_tier(index, future_label);
      if (future_tier == 1) {
        std::shared_ptr<uint32_t> host_graph_storage;
        {
          std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
          auto cache_it = index.phoenix_label_cache->host_graphs.find(future_label);
          if (cache_it != index.phoenix_label_cache->host_graphs.end()) {
            index.phoenix_label_cache->host_lru_labels.splice(
              index.phoenix_label_cache->host_lru_labels.begin(),
              index.phoenix_label_cache->host_lru_labels,
              cache_it->second.lru_it);
            host_graph_storage = cache_it->second.storage;
          }
        }
        if (host_graph_storage == nullptr) { continue; }
        prefetched_dram_request = std::make_unique<async_host_graph_copy_request>(
          std::move(host_graph_storage),
          future_requested_bytes,
          future_label,
          future_label_offset,
          future_label_size);
        prefetched_label = future_label;
        return;
      }
      if (future_tier == 2) {
        prefetched_ssd_request =
          std::make_unique<detail::phoenix::async_ibin_graph_rows_request>(
            index.cagra_graph_cache_fname,
            total_graph_rows,
            cached_graph_storage_width(index),
            future_label_offset,
            future_label_size);
        prefetched_label = future_label;
        return;
      }
    }
  };

  for (std::size_t label_position = 0; label_position < serial_labels.size(); ++label_position) {
    auto label = serial_labels[label_position];
    if (label >= index.host_cagra_label_size.size()) {
      throw std::runtime_error("Phoenix label-load query label is out of bounds: " +
                               std::to_string(label));
    }

    auto label_size = static_cast<int64_t>(index.host_cagra_label_size[label]);
    auto label_offset = static_cast<int64_t>(index.host_cagra_label_offset[label]);
    if (label_size <= 0) {
      throw std::runtime_error("Phoenix label-load received a CAGRA query for non-CAGRA label " +
                               std::to_string(label));
    }
    record_phoenix_label_access(res, index, label, query_dim);

    auto const& host_positions = query_positions_by_label.at(label);
    auto group_size = static_cast<int64_t>(host_positions.size());
    auto scratch_positions = raft::make_device_vector<uint32_t, int64_t>(res, group_size);
    auto scratch_queries = raft::make_device_matrix<data_t, int64_t>(res, group_size, query_dim);
    auto scratch_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, group_size);
    auto scratch_index_map = raft::make_device_vector<uint32_t, int64_t>(res, label_size);
    auto scratch_label_size = raft::make_device_vector<uint32_t, int64_t>(res, 1);
    auto scratch_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, 1);
    auto scratch_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, group_size, topk);
    auto scratch_distances = raft::make_device_matrix<float, int64_t>(res, group_size, topk);
    uint32_t host_group_label_offset = 0;
    raft::update_device(
      scratch_label_offset.data_handle(), &host_group_label_offset, 1, stream);

    raft::update_device(
      scratch_positions.data_handle(), host_positions.data(), group_size, stream);
    auto total_values = group_size * query_dim;
    auto block_size = 256;
    if (total_values > 0) {
      auto grid_size = static_cast<int>((total_values + block_size - 1) / block_size);
      gather_rows_kernel<<<grid_size, block_size, 0, stream>>>(query_info.cagra_queries.data_handle(),
                                                               scratch_positions.data_handle(),
                                                               scratch_queries.data_handle(),
                                                               group_size,
                                                               query_dim);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    RAFT_CUDA_TRY(cudaMemsetAsync(
      scratch_query_labels.data_handle(), 0, group_size * sizeof(uint32_t), stream));
    raft::copy(scratch_index_map.data_handle(),
               index.cagra_index_map.data_handle() + label_offset,
               label_size,
               stream);
    uint32_t host_group_label_size = static_cast<uint32_t>(label_size);
    raft::update_device(scratch_label_size.data_handle(), &host_group_label_size, 1, stream);

    auto group_queries_view =
      raft::make_device_matrix_view<const data_t, int64_t, raft::row_major>(
        scratch_queries.data_handle(), group_size, query_dim);
    auto group_query_labels_view = raft::make_device_vector_view<uint32_t, int64_t>(
      scratch_query_labels.data_handle(), group_size);
    auto group_index_map_view = raft::make_device_vector_view<uint32_t, int64_t>(
      scratch_index_map.data_handle(), label_size);
    auto group_label_size_view = raft::make_device_vector_view<uint32_t, int64_t>(
      scratch_label_size.data_handle(), 1);
    auto group_label_offset_view = raft::make_device_vector_view<uint32_t, int64_t>(
      scratch_label_offset.data_handle(), 1);
    auto group_neighbors_view =
      raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
        scratch_neighbors.data_handle(), group_size, topk);
    auto group_distances_view =
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(
        scratch_distances.data_handle(), group_size, topk);
    std::shared_ptr<data_t> dataset_storage;

    std::shared_ptr<uint32_t> graph_storage;
    if (prefetched_dram_request != nullptr && prefetched_label == label) {
      vecflow_log("Using prefetched cached host graph rows [",
                  label_offset,
                  ", ",
                  (label_offset + label_size),
                  ") for label ",
                  label);
      graph_storage = prefetched_dram_request->wait_and_release();
      prefetched_dram_request.reset();
      prefetched_label = UINT32_MAX;
      index.phoenix_label_cache->dram_hits += 1;
      graph_storage =
        finalize_loaded_phoenix_label_graph(res, index, label, label_offset, label_size,
                                            std::move(graph_storage));
    } else if (prefetched_ssd_request != nullptr && prefetched_label == label) {
      vecflow_log("Using prefetched graph rows [",
                  label_offset,
                  ", ",
                  (label_offset + label_size),
                  ") for label ",
                  label);
      graph_storage = prefetched_ssd_request->wait_and_release();
      prefetched_ssd_request.reset();
      prefetched_label = UINT32_MAX;
      index.phoenix_label_cache->ssd_loads += 1;
      graph_storage =
        finalize_loaded_phoenix_label_graph(res, index, label, label_offset, label_size,
                                            std::move(graph_storage));
    } else {
      graph_storage =
        get_or_load_phoenix_label_graph(res, index, label, total_graph_rows, label_offset, label_size);
    }
    schedule_future_prefetch(label_position);
    if (use_dataset_label_load) {
      schedule_future_dataset_prefetch<data_t>(index,
                                               serial_labels,
                                               label_position,
                                               total_graph_rows,
                                               query_dim,
                                               prefetched_dataset_dram_request,
                                               prefetched_dataset_ssd_request,
                                               &prefetched_dataset_label);
    }

    auto graph_view = raft::make_device_matrix_view<const uint32_t, int64_t, raft::row_major>(
      graph_storage.get(), label_size, cached_graph_storage_width(index));

    auto label_index =
      cuvs::neighbors::cagra::index<data_t, uint32_t>(res, index.ivf_graph_index.metric());
    if (use_dataset_label_load) {
      dataset_storage = resolve_prefetched_or_load_phoenix_label_dataset<data_t>(
        res,
        index,
        label,
        total_graph_rows,
        label_offset,
        label_size,
        query_dim,
        prefetched_dataset_dram_request,
        prefetched_dataset_ssd_request,
        &prefetched_dataset_label);
      auto dataset_view = raft::make_device_matrix_view<const data_t, int64_t, raft::row_major>(
        dataset_storage.get(), label_size, query_dim);
      label_index.update_dataset(res, dataset_view);
    } else {
      label_index.update_dataset(res, index.ivf_graph_index.dataset());
    }
    label_index.update_graph(res, graph_view);

    cagra::filtered_search(
      res,
      search_params,
      label_index,
      group_queries_view,
      group_neighbors_view,
      group_distances_view,
      group_query_labels_view,
      group_index_map_view,
      group_label_size_view,
      group_label_offset_view,
      sample_filter);

    auto total_results = group_size * topk;
    auto result_grid = static_cast<int>((total_results + block_size - 1) / block_size);
    merge_neighbors_kernel<data_t, uint32_t><<<result_grid, block_size, 0, stream>>>(
      cagra_neighbors.data_handle(),
      cagra_distances.data_handle(),
      scratch_neighbors.data_handle(),
      scratch_distances.data_handle(),
      scratch_positions.data_handle(),
      group_size,
      topk);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
#endif
}

template<typename data_t>
void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<data_t>& index,
            raft::device_matrix_view<const data_t, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{}) {

  initialize_search_results(res, neighbors, distances);
	
	// Configure standard search parameters
  cuvs::neighbors::cagra::search_params search_params;
  search_params.algo = cagra::search_algo::SINGLE_CTA_FILTERED;
	search_params.itopk_size = itopk_size;

  auto stream = raft::resource::get_cuda_stream(res);
  if (index.query_classification_scratch == nullptr) {
    index.query_classification_scratch =
      std::make_shared<query_classification_scratch<data_t>>(stream);
  }

  auto query_info = classify_queries<data_t>(res,
																						 queries,
																						 query_labels,
																						 index.cat_freq.view(),
																						 index.specificity_threshold,
                                             index.query_classification_scratch.get());

  int64_t topk = neighbors.extent(1);
  int n_cagra_queries = query_info.cagra_query_map.size();
  int n_bfs_queries = query_info.bfs_query_map.size();
  auto cagra_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_cagra_queries, topk);
  auto cagra_distances = raft::make_device_matrix<float, int64_t>(res, n_cagra_queries, topk);
  auto bfs_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_bfs_queries, topk);
  auto bfs_distances = raft::make_device_matrix<float, int64_t>(res, n_bfs_queries, topk);
  raft::resource::sync_stream(res);

  if (n_cagra_queries > 0) {
    auto use_label_load =
      detail::phoenix::use_phoenix_label_load() && index.ivf_graph_index.graph().extent(0) == 0;
    if (use_label_load) {
      search_cagra_with_phoenix_label_load(res,
                                          index,
                                          search_params,
                                          query_info,
                                          cagra_neighbors.view(),
                                          cagra_distances.view(),
                                          topk,
                                          sample_filter);
    } else {
      cagra::filtered_search(res,
                             search_params,
                             index.ivf_graph_index,
                             query_info.cagra_queries.view(),
                             cagra_neighbors.view(),
                             cagra_distances.view(),
                             query_info.cagra_query_labels.view(),
                             index.cagra_index_map.view(),
                             index.cagra_label_size.view(),
                             index.cagra_label_offset.view(),
                             sample_filter);
    }
    raft::resource::sync_stream(res);
  }

  if (n_bfs_queries > 0) {
    bool use_direct_bfs = !index.bfs_tiered_cache_enabled ||
                          index.ivf_bfs_index.n_lists() > 0;
    if (!use_direct_bfs) {
      search_bfs_with_tiered_cache(
        res, index, query_info, bfs_neighbors.view(), bfs_distances.view(), topk, sample_filter);
    } else {
      search_filtered_bfs(res,
                          index.ivf_bfs_index,
                          raft::make_const_mdspan(query_info.bfs_queries.view()),
                          query_info.bfs_query_labels.view(),
                          index.bfs_label_size.view(),
                          bfs_neighbors.view(),
                          bfs_distances.view(),
                          cuvs::distance::DistanceType::L2Unexpanded,
                          sample_filter);
    }
    raft::resource::sync_stream(res);
  }
  
  merge_search_results<data_t>(res,
															 neighbors,
															 distances,
															 query_info,
															 bfs_neighbors.view(),
															 bfs_distances.view(),
															 cagra_neighbors.view(),
															 cagra_distances.view(),
															 topk);
}


struct host_multi_label_query_desc {
  std::vector<int64_t> label_offsets;
  std::vector<uint32_t> label_indices;
};

inline auto copy_multi_label_query_desc_to_host(shared_resources::configured_raft_resources& res,
                                                const multi_label_query_desc& desc)
  -> host_multi_label_query_desc
{
  host_multi_label_query_desc host_desc;
  if (!desc.host_label_offsets.empty() || !desc.host_label_indices.empty()) {
    host_desc.label_offsets = desc.host_label_offsets;
    host_desc.label_indices = desc.host_label_indices;
    return host_desc;
  }
  host_desc.label_offsets.resize(static_cast<std::size_t>(desc.label_offsets.extent(0)));
  host_desc.label_indices.resize(static_cast<std::size_t>(desc.label_indices.extent(0)));
  auto stream = raft::resource::get_cuda_stream(res);
  if (!host_desc.label_offsets.empty()) {
    raft::copy(host_desc.label_offsets.data(),
               desc.label_offsets.data_handle(),
               static_cast<int64_t>(host_desc.label_offsets.size()),
               stream);
  }
  if (!host_desc.label_indices.empty()) {
    raft::copy(host_desc.label_indices.data(),
               desc.label_indices.data_handle(),
               static_cast<int64_t>(host_desc.label_indices.size()),
               stream);
  }
  raft::resource::sync_stream(res);
  return host_desc;
}

template <typename data_t>
inline auto label_has_members(const cuvs::neighbors::vecflow::index<data_t>& index, uint32_t label)
  -> bool
{
  auto cagra_has_members =
    label < index.host_cagra_label_size.size() && index.host_cagra_label_size[label] > 0;
  auto bfs_has_members =
    label < index.host_bfs_label_size.size() && index.host_bfs_label_size[label] > 0;
  return cagra_has_members || bfs_has_members;
}

inline auto normalized_query_labels_for_query(const host_multi_label_query_desc& query_desc,
                                              uint32_t query_id) -> std::vector<uint32_t>
{
  if (query_id + 1 >= query_desc.label_offsets.size()) { return {}; }
  auto start = query_desc.label_offsets[query_id];
  auto end   = query_desc.label_offsets[query_id + 1];
  if (start < 0 || end < start || static_cast<std::size_t>(end) > query_desc.label_indices.size()) {
    throw std::runtime_error("Multi-label query CSR offsets are invalid");
  }

  std::vector<uint32_t> labels(query_desc.label_indices.begin() + start,
                               query_desc.label_indices.begin() + end);
  std::sort(labels.begin(), labels.end());
  labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
  return labels;
}

struct normalized_multi_label_query_data {
  std::vector<std::vector<uint32_t>> labels_by_query;
  std::vector<int64_t> label_offsets;
  std::vector<uint32_t> label_indices;
};

template <typename data_t>
inline auto normalize_query_labels_for_search(const cuvs::neighbors::vecflow::index<data_t>& index,
                                              const host_multi_label_query_desc& query_desc)
  -> normalized_multi_label_query_data
{
  auto query_count = query_desc.label_offsets.empty() ? std::size_t{0}
                                                      : (query_desc.label_offsets.size() - 1);
  normalized_multi_label_query_data normalized;
  normalized.labels_by_query.resize(query_count);
  normalized.label_offsets.reserve(query_count + 1);
  normalized.label_offsets.push_back(0);

  for (uint32_t query = 0; query < static_cast<uint32_t>(query_count); ++query) {
    auto labels = normalized_query_labels_for_query(query_desc, query);
    labels.erase(std::remove_if(labels.begin(),
                                labels.end(),
                                [&](uint32_t label) { return !label_has_members(index, label); }),
                 labels.end());
    normalized.labels_by_query[query] = labels;
    normalized.label_indices.insert(
      normalized.label_indices.end(), labels.begin(), labels.end());
    normalized.label_offsets.push_back(static_cast<int64_t>(normalized.label_indices.size()));
  }

  return normalized;
}

struct invalid_sample_tuple_predicate {
  template <typename Tuple>
  __host__ __device__ bool operator()(Tuple const& value) const
  {
    return thrust::get<1>(value) == UINT32_MAX;
  }
};

struct query_distance_sample_less {
  template <typename LhsTuple, typename RhsTuple>
  __host__ __device__ bool operator()(LhsTuple const& lhs, RhsTuple const& rhs) const
  {
    auto lhs_query = thrust::get<0>(lhs);
    auto rhs_query = thrust::get<0>(rhs);
    if (lhs_query != rhs_query) { return lhs_query < rhs_query; }
    auto lhs_distance = thrust::get<2>(lhs);
    auto rhs_distance = thrust::get<2>(rhs);
    if (lhs_distance != rhs_distance) { return lhs_distance < rhs_distance; }
    return thrust::get<1>(lhs) < thrust::get<1>(rhs);
  }
};

inline auto count_valid_results_in_output(shared_resources::configured_raft_resources& res,
                                          raft::device_matrix_view<uint32_t, int64_t> neighbors)
  -> std::vector<uint32_t>
{
  std::vector<uint32_t> host_counts(static_cast<std::size_t>(neighbors.extent(0)), 0);
  if (neighbors.extent(0) == 0 || neighbors.extent(1) == 0) { return host_counts; }

  auto stream = raft::resource::get_cuda_stream(res);
  auto d_counts = raft::make_device_vector<uint32_t, int64_t>(res, neighbors.extent(0));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    d_counts.data_handle(), 0, neighbors.extent(0) * sizeof(uint32_t), stream));

  auto total = neighbors.extent(0) * neighbors.extent(1);
  auto block_size = 256;
  auto grid_size = static_cast<int>((total + block_size - 1) / block_size);
  count_valid_results_kernel<<<grid_size, block_size, 0, stream>>>(neighbors.data_handle(),
                                                                   neighbors.extent(0),
                                                                   static_cast<int>(neighbors.extent(1)),
                                                                   d_counts.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::copy(host_counts.data(), d_counts.data_handle(), neighbors.extent(0), stream);
  raft::resource::sync_stream(res);
  return host_counts;
}

template <typename data_t>
inline void filter_device_results_by_query_membership(
  shared_resources::configured_raft_resources& res,
  cuvs::neighbors::vecflow::index<data_t>& index,
  raft::device_vector_view<uint32_t, int64_t> query_label_indices,
  raft::device_vector_view<int64_t, int64_t> query_label_offsets,
  raft::device_vector_view<uint32_t, int64_t> subquery_to_query,
  raft::device_matrix_view<uint32_t, int64_t> neighbors,
  raft::device_matrix_view<float, int64_t> distances,
  int64_t query_count)
{
  if (subquery_to_query.extent(0) == 0 || neighbors.extent(1) == 0 || query_count == 0) { return; }

  auto stream = raft::resource::get_cuda_stream(res);
  auto d_valid_counts = raft::make_device_vector<uint32_t, int64_t>(res, query_count);
  RAFT_CUDA_TRY(cudaMemsetAsync(
    d_valid_counts.data_handle(), 0, query_count * sizeof(uint32_t), stream));

  auto total = subquery_to_query.extent(0) * neighbors.extent(1);
  auto block_size = 256;
  auto grid_size = static_cast<int>((total + block_size - 1) / block_size);
  filter_results_by_membership_kernel<<<grid_size, block_size, 0, stream>>>(
    query_label_offsets.data_handle(),
    query_label_indices.data_handle(),
    subquery_to_query.data_handle(),
    index.cagra_label_size.data_handle(),
    index.cagra_label_offset.data_handle(),
    index.cagra_index_map.data_handle(),
    index.bfs_label_size.data_handle(),
    index.bfs_label_offset.data_handle(),
    index.bfs_index_map.data_handle(),
    neighbors.data_handle(),
    distances.data_handle(),
    subquery_to_query.extent(0),
    static_cast<int>(neighbors.extent(1)),
    d_valid_counts.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

constexpr int64_t kMergeDeviceResultsChunkEntries = int64_t{4} << 20;

inline void merge_device_results_chunk_to_output(
  shared_resources::configured_raft_resources& res,
  raft::device_vector_view<uint32_t, int64_t> batch_query_ids,
  raft::device_matrix_view<uint32_t, int64_t> batch_neighbors,
  raft::device_matrix_view<float, int64_t> batch_distances,
  uint32_t query_id_base,
  int64_t output_queries,
  int topk,
  raft::device_matrix_view<uint32_t, int64_t> neighbors,
  raft::device_matrix_view<float, int64_t> distances,
  bool include_existing_output)
{
  auto stream = raft::resource::get_cuda_stream(res);
  auto batch_entries = batch_query_ids.extent(0) * batch_neighbors.extent(1);
  auto existing_entries = include_existing_output ? output_queries * neighbors.extent(1) : int64_t{0};
  auto total_entries = batch_entries + existing_entries;
  if (total_entries == 0) {
    initialize_search_results(res, neighbors, distances);
    return;
  }

  rmm::device_uvector<uint32_t> flat_query_ids(total_entries, stream);
  rmm::device_uvector<uint32_t> flat_sample_ids(total_entries, stream);
  rmm::device_uvector<float> flat_distances(total_entries, stream);

  auto block_size = 256;
  if (batch_entries > 0) {
    auto grid_size = static_cast<int>((batch_entries + block_size - 1) / block_size);
    flatten_subquery_results_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_query_ids.data_handle(),
      batch_neighbors.data_handle(),
      batch_distances.data_handle(),
      flat_query_ids.data(),
      flat_sample_ids.data(),
      flat_distances.data(),
      query_id_base,
      batch_query_ids.extent(0),
      static_cast<int>(batch_neighbors.extent(1)));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  if (existing_entries > 0) {
    auto grid_size = static_cast<int>((existing_entries + block_size - 1) / block_size);
    flatten_output_results_kernel<<<grid_size, block_size, 0, stream>>>(
      neighbors.data_handle(),
      distances.data_handle(),
      flat_query_ids.data() + batch_entries,
      flat_sample_ids.data() + batch_entries,
      flat_distances.data() + batch_entries,
      output_queries,
      static_cast<int>(neighbors.extent(1)));
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  auto zip_begin = thrust::make_zip_iterator(
    thrust::make_tuple(flat_query_ids.begin(), flat_sample_ids.begin(), flat_distances.begin()));
  auto zip_end = zip_begin + total_entries;
  auto compact_end = thrust::remove_if(raft::resource::get_thrust_policy(res),
                                       zip_begin,
                                       zip_end,
                                       invalid_sample_tuple_predicate{});
  auto valid_entries = static_cast<int64_t>(compact_end - zip_begin);
  if (valid_entries == 0) {
    initialize_search_results(res, neighbors, distances);
    return;
  }

  rmm::device_uvector<uint64_t> pair_keys(valid_entries, stream);
  auto key_grid = static_cast<int>((valid_entries + block_size - 1) / block_size);
  pack_query_sample_keys_kernel<<<key_grid, block_size, 0, stream>>>(
    flat_query_ids.data(), flat_sample_ids.data(), pair_keys.data(), valid_entries);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  thrust::sort_by_key(raft::resource::get_thrust_policy(res),
                      pair_keys.begin(),
                      pair_keys.end(),
                      flat_distances.begin());

  rmm::device_uvector<uint64_t> reduced_keys(valid_entries, stream);
  rmm::device_uvector<float> reduced_distances(valid_entries, stream);
  auto reduced = thrust::reduce_by_key(raft::resource::get_thrust_policy(res),
                                       pair_keys.begin(),
                                       pair_keys.end(),
                                       flat_distances.begin(),
                                       reduced_keys.begin(),
                                       reduced_distances.begin(),
                                       thrust::equal_to<uint64_t>{},
                                       thrust::minimum<float>{});
  auto unique_entries = static_cast<int64_t>(reduced.first - reduced_keys.begin());
  if (unique_entries == 0) { return; }

  auto unique_grid = static_cast<int>((unique_entries + block_size - 1) / block_size);
  unpack_query_sample_keys_kernel<<<unique_grid, block_size, 0, stream>>>(
    reduced_keys.data(), flat_query_ids.data(), flat_sample_ids.data(), unique_entries);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto sort_begin = thrust::make_zip_iterator(
    thrust::make_tuple(flat_query_ids.begin(), flat_sample_ids.begin(), reduced_distances.begin()));
  thrust::sort(raft::resource::get_thrust_policy(res),
               sort_begin,
               sort_begin + unique_entries,
               query_distance_sample_less{});

  rmm::device_uvector<int64_t> query_offsets(output_queries + 1, stream);
  thrust::lower_bound(raft::resource::get_thrust_policy(res),
                      flat_query_ids.begin(),
                      flat_query_ids.begin() + unique_entries,
                      thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(output_queries + 1)),
                      query_offsets.begin());

  auto scatter_total = output_queries * static_cast<int64_t>(topk);
  initialize_search_results(res, neighbors, distances);
  if (scatter_total == 0) { return; }
  auto scatter_grid = static_cast<int>((scatter_total + block_size - 1) / block_size);
  scatter_sorted_topk_kernel<<<scatter_grid, block_size, 0, stream>>>(query_offsets.data(),
                                                                      flat_sample_ids.data(),
                                                                      reduced_distances.data(),
                                                                      output_queries,
                                                                      topk,
                                                                      neighbors.data_handle(),
                                                                      distances.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

inline void merge_device_results_to_output(shared_resources::configured_raft_resources& res,
                                           raft::device_vector_view<uint32_t, int64_t> batch_query_ids,
                                           raft::device_matrix_view<uint32_t, int64_t> batch_neighbors,
                                           raft::device_matrix_view<float, int64_t> batch_distances,
                                           int64_t output_queries,
                                           int topk,
                                           raft::device_matrix_view<uint32_t, int64_t> neighbors,
                                           raft::device_matrix_view<float, int64_t> distances,
                                           bool include_existing_output)
{
  auto batch_entries = batch_query_ids.extent(0) * batch_neighbors.extent(1);
  auto existing_entries = include_existing_output ? output_queries * neighbors.extent(1) : int64_t{0};
  auto total_entries = batch_entries + existing_entries;
  if (total_entries == 0) {
    initialize_search_results(res, neighbors, distances);
    return;
  }

  if (total_entries <= kMergeDeviceResultsChunkEntries || batch_query_ids.extent(0) == 0 ||
      output_queries <= 1) {
    merge_device_results_chunk_to_output(res,
                                         batch_query_ids,
                                         batch_neighbors,
                                         batch_distances,
                                         0,
                                         output_queries,
                                         topk,
                                         neighbors,
                                         distances,
                                         include_existing_output);
    return;
  }

  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<uint32_t> host_batch_query_ids(static_cast<std::size_t>(batch_query_ids.extent(0)));
  raft::copy(host_batch_query_ids.data(),
             batch_query_ids.data_handle(),
             batch_query_ids.extent(0),
             stream);
  raft::resource::sync_stream(res);

  if (!std::is_sorted(host_batch_query_ids.begin(), host_batch_query_ids.end())) {
    vecflow_log("WARNING: merge_device_results_to_output received unsorted query_ids (",
                host_batch_query_ids.size(),
                " entries, ", total_entries,
                " total). Falling back to unchunked merge; OOM risk for large batches.");
    merge_device_results_chunk_to_output(res,
                                         batch_query_ids,
                                         batch_neighbors,
                                         batch_distances,
                                         0,
                                         output_queries,
                                         topk,
                                         neighbors,
                                         distances,
                                         include_existing_output);
    return;
  }

  std::vector<int64_t> batch_query_offsets(static_cast<std::size_t>(output_queries + 1), 0);
  auto cursor = int64_t{0};
  for (int64_t query = 0; query < output_queries; ++query) {
    batch_query_offsets[static_cast<std::size_t>(query)] = cursor;
    while (cursor < batch_query_ids.extent(0) &&
           host_batch_query_ids[static_cast<std::size_t>(cursor)] == static_cast<uint32_t>(query)) {
      ++cursor;
    }
  }
  batch_query_offsets[static_cast<std::size_t>(output_queries)] = cursor;
  if (cursor != batch_query_ids.extent(0)) {
    merge_device_results_chunk_to_output(res,
                                         batch_query_ids,
                                         batch_neighbors,
                                         batch_distances,
                                         0,
                                         output_queries,
                                         topk,
                                         neighbors,
                                         distances,
                                         include_existing_output);
    return;
  }

  auto batch_topk = batch_neighbors.extent(1);
  auto output_topk = neighbors.extent(1);
  for (int64_t chunk_start = 0; chunk_start < output_queries;) {
    auto chunk_end = chunk_start;
    while (chunk_end < output_queries) {
      auto next_end = chunk_end + 1;
      auto chunk_queries = next_end - chunk_start;
      auto chunk_subqueries = batch_query_offsets[static_cast<std::size_t>(next_end)] -
                              batch_query_offsets[static_cast<std::size_t>(chunk_start)];
      auto chunk_total_entries =
        chunk_subqueries * batch_topk +
        (include_existing_output ? chunk_queries * output_topk : int64_t{0});
      if (chunk_end > chunk_start && chunk_total_entries > kMergeDeviceResultsChunkEntries) { break; }
      chunk_end = next_end;
      if (chunk_total_entries >= kMergeDeviceResultsChunkEntries) { break; }
    }

    auto chunk_query_count = chunk_end - chunk_start;
    auto subquery_begin = batch_query_offsets[static_cast<std::size_t>(chunk_start)];
    auto subquery_end = batch_query_offsets[static_cast<std::size_t>(chunk_end)];
    auto chunk_subquery_count = subquery_end - subquery_begin;

    auto chunk_batch_query_ids = raft::make_device_vector_view<uint32_t, int64_t>(
      batch_query_ids.data_handle() + subquery_begin, chunk_subquery_count);
    auto chunk_batch_neighbors = raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
      batch_neighbors.data_handle() + subquery_begin * batch_topk, chunk_subquery_count, batch_topk);
    auto chunk_batch_distances = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
      batch_distances.data_handle() + subquery_begin * batch_topk, chunk_subquery_count, batch_topk);
    auto chunk_neighbors = raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
      neighbors.data_handle() + chunk_start * output_topk, chunk_query_count, output_topk);
    auto chunk_distances = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
      distances.data_handle() + chunk_start * output_topk, chunk_query_count, output_topk);

    merge_device_results_chunk_to_output(res,
                                         chunk_batch_query_ids,
                                         chunk_batch_neighbors,
                                         chunk_batch_distances,
                                         static_cast<uint32_t>(chunk_start),
                                         chunk_query_count,
                                         topk,
                                         chunk_neighbors,
                                         chunk_distances,
                                         include_existing_output);
    chunk_start = chunk_end;
  }
}

template <typename data_t>
inline auto copy_cat_freq_to_host(shared_resources::configured_raft_resources& res,
                                  cuvs::neighbors::vecflow::index<data_t>& index)
  -> const std::vector<uint32_t>&
{
  if (index.host_cat_freq.empty() && index.cat_freq.extent(0) > 0) {
    index.host_cat_freq.resize(static_cast<std::size_t>(index.cat_freq.extent(0)));
    raft::copy(index.host_cat_freq.data(),
               index.cat_freq.data_handle(),
               static_cast<int64_t>(index.host_cat_freq.size()),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
  }
  return index.host_cat_freq;
}

template <typename data_t>
inline auto load_host_label_members(shared_resources::configured_raft_resources& res,
                                    cuvs::neighbors::vecflow::index<data_t>& index,
                                    uint32_t label,
                                    std::unordered_map<uint32_t, std::vector<uint32_t>>& cache)
  -> const std::vector<uint32_t>&
{
  auto it = cache.find(label);
  if (it != cache.end()) { return it->second; }

  std::vector<uint32_t> values;
  if (label < index.host_bfs_label_size.size() && index.host_bfs_label_size[label] > 0) {
    auto offset = static_cast<std::size_t>(index.host_bfs_label_offset[label]);
    auto size   = static_cast<std::size_t>(index.host_bfs_label_size[label]);
    values.assign(index.host_bfs_index_map.begin() + offset,
                  index.host_bfs_index_map.begin() + offset + size);
  } else if (label < index.host_cagra_label_size.size() && index.host_cagra_label_size[label] > 0) {
    auto size   = static_cast<std::size_t>(index.host_cagra_label_size[label]);
    auto offset = static_cast<int64_t>(index.host_cagra_label_offset[label]);
    values.resize(size);
    raft::copy(values.data(),
               index.cagra_index_map.data_handle() + offset,
               static_cast<int64_t>(size),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
  }

  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());

  auto [inserted_it, _] = cache.emplace(label, std::move(values));
  return inserted_it->second;
}

template <typename data_t>
inline auto build_and_candidate_lists(shared_resources::configured_raft_resources& res,
                                      cuvs::neighbors::vecflow::index<data_t>& index,
                                      const host_multi_label_query_desc& query_desc)
  -> std::vector<std::vector<uint32_t>>
{
  auto query_count = query_desc.label_offsets.empty() ? std::size_t{0}
                                                      : (query_desc.label_offsets.size() - 1);
  std::vector<std::vector<uint32_t>> candidates_by_query(query_count);
  std::unordered_map<uint32_t, std::vector<uint32_t>> label_members_cache;
  label_members_cache.reserve(query_count * 2);

  for (uint32_t query = 0; query < static_cast<uint32_t>(query_count); ++query) {
    auto labels = normalized_query_labels_for_query(query_desc, query);
    labels.erase(std::remove_if(labels.begin(),
                                labels.end(),
                                [&](uint32_t label) { return !label_has_members(index, label); }),
                 labels.end());
    if (labels.empty()) { continue; }

    std::vector<uint32_t> matches;
    bool initialized = false;
    for (auto label : labels) {
      auto const& label_members = load_host_label_members(res, index, label, label_members_cache);
      if (!initialized) {
        matches = label_members;
        initialized = true;
      } else {
        std::vector<uint32_t> intersection;
        intersection.reserve(std::min(matches.size(), label_members.size()));
        std::set_intersection(matches.begin(),
                              matches.end(),
                              label_members.begin(),
                              label_members.end(),
                              std::back_inserter(intersection));
        matches = std::move(intersection);
      }
      if (matches.empty()) { break; }
    }
    candidates_by_query[query] = std::move(matches);
  }

  return candidates_by_query;
}

template <typename data_t>
inline auto gather_queries(shared_resources::configured_raft_resources& res,
                           raft::device_matrix_view<const data_t, int64_t> queries,
                           const std::vector<uint32_t>& query_ids)
  -> raft::device_matrix<data_t, int64_t>
{
  auto gathered = raft::make_device_matrix<data_t, int64_t>(
    res, static_cast<int64_t>(query_ids.size()), queries.extent(1));
  if (query_ids.empty()) { return gathered; }

  auto d_query_ids =
    raft::make_device_vector<uint32_t, int64_t>(res, static_cast<int64_t>(query_ids.size()));
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(d_query_ids.data_handle(),
                      query_ids.data(),
                      static_cast<int64_t>(query_ids.size()),
                      stream);
  auto total_values = static_cast<int64_t>(query_ids.size()) * queries.extent(1);
  auto block_size   = 256;
  auto grid_size    = static_cast<int>((total_values + block_size - 1) / block_size);
  gather_rows_kernel<<<grid_size, block_size, 0, stream>>>(queries.data_handle(),
                                                           d_query_ids.data_handle(),
                                                           gathered.data_handle(),
                                                           static_cast<int64_t>(query_ids.size()),
                                                           queries.extent(1));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(res);
  return gathered;
}

inline auto filter_subquery_results_by_allowed(
  const std::vector<uint32_t>& subquery_to_query,
  std::vector<uint32_t>* sub_neighbors,
  std::vector<float>* sub_distances,
  int subquery_topk,
  const std::vector<std::vector<uint32_t>>& allowed_by_query) -> std::vector<int>
{
  std::vector<int> valid_counts(allowed_by_query.size(), 0);
  if (sub_neighbors == nullptr || sub_distances == nullptr) { return valid_counts; }

  for (std::size_t subquery = 0; subquery < subquery_to_query.size(); ++subquery) {
    auto query_id = subquery_to_query[subquery];
    if (query_id >= allowed_by_query.size()) {
      throw std::runtime_error("Expanded multi-label query id is out of bounds during AND filtering");
    }
    auto const& allowed = allowed_by_query[query_id];
    auto base = subquery * static_cast<std::size_t>(subquery_topk);
    for (int k = 0; k < subquery_topk; ++k) {
      auto idx = base + static_cast<std::size_t>(k);
      auto sample_id = (*sub_neighbors)[idx];
      if (sample_id == UINT32_MAX) { continue; }
      if (allowed.empty() ||
          !std::binary_search(allowed.begin(), allowed.end(), sample_id)) {
        (*sub_neighbors)[idx] = UINT32_MAX;
        (*sub_distances)[idx] = std::numeric_limits<float>::infinity();
        continue;
      }
      valid_counts[query_id] += 1;
    }
  }

  return valid_counts;
}

inline void merge_subquery_results_to_output(shared_resources::configured_raft_resources& res,
                                             const std::vector<uint32_t>& subquery_to_query,
                                             const std::vector<uint32_t>& sub_neighbors,
                                             const std::vector<float>& sub_distances,
                                             int subquery_topk,
                                             int64_t output_queries,
                                             int topk,
                                             raft::device_matrix_view<uint32_t, int64_t> neighbors,
                                             raft::device_matrix_view<float, int64_t> distances)
{
  std::vector<uint32_t> host_neighbors(static_cast<std::size_t>(output_queries) * topk, UINT32_MAX);
  std::vector<float> host_distances(static_cast<std::size_t>(output_queries) * topk,
                                    std::numeric_limits<float>::infinity());
  std::vector<std::unordered_map<uint32_t, float>> candidates(static_cast<std::size_t>(output_queries));

  for (std::size_t subquery = 0; subquery < subquery_to_query.size(); ++subquery) {
    auto query_id = subquery_to_query[subquery];
    if (query_id >= static_cast<uint32_t>(output_queries)) {
      throw std::runtime_error("Expanded multi-label query id is out of bounds");
    }
    auto& query_candidates = candidates[query_id];
    auto base = subquery * static_cast<std::size_t>(subquery_topk);
    for (int k = 0; k < subquery_topk; ++k) {
      auto sample_id = sub_neighbors[base + static_cast<std::size_t>(k)];
      if (sample_id == UINT32_MAX) { continue; }
      auto distance = sub_distances[base + static_cast<std::size_t>(k)];
      auto [it, inserted] = query_candidates.try_emplace(sample_id, distance);
      if (!inserted && distance < it->second) { it->second = distance; }
    }
  }

  for (int64_t query = 0; query < output_queries; ++query) {
    std::vector<std::pair<uint32_t, float>> sorted_candidates;
    auto& query_candidates = candidates[static_cast<std::size_t>(query)];
    sorted_candidates.reserve(query_candidates.size());
    for (auto const& entry : query_candidates) { sorted_candidates.push_back(entry); }
    std::sort(sorted_candidates.begin(),
              sorted_candidates.end(),
              [](auto const& lhs, auto const& rhs) {
                if (lhs.second != rhs.second) { return lhs.second < rhs.second; }
                return lhs.first < rhs.first;
              });

    auto limit = std::min<int>(topk, static_cast<int>(sorted_candidates.size()));
    auto base = static_cast<std::size_t>(query) * topk;
    for (int k = 0; k < limit; ++k) {
      host_neighbors[base + static_cast<std::size_t>(k)] = sorted_candidates[k].first;
      host_distances[base + static_cast<std::size_t>(k)] = sorted_candidates[k].second;
    }
  }

  if (!host_neighbors.empty()) {
    auto stream = raft::resource::get_cuda_stream(res);
    raft::copy(neighbors.data_handle(),
               host_neighbors.data(),
               static_cast<int64_t>(host_neighbors.size()),
               stream);
    raft::copy(distances.data_handle(),
               host_distances.data(),
               static_cast<int64_t>(host_distances.size()),
               stream);
    raft::resource::sync_stream(res);
  }
}

template <typename data_t>
inline void search_multi_label_or(shared_resources::configured_raft_resources& res,
                                  cuvs::neighbors::vecflow::index<data_t>& index,
                                  raft::device_matrix_view<const data_t, int64_t> queries,
                                  const host_multi_label_query_desc& query_desc,
                                  int itopk_size,
                                  raft::device_matrix_view<uint32_t, int64_t> neighbors,
                                  raft::device_matrix_view<float, int64_t> distances)
{
  auto normalized_queries = normalize_query_labels_for_search(index, query_desc);
  std::vector<uint32_t> expanded_query_ids;
  std::vector<uint32_t> expanded_labels;
  expanded_query_ids.reserve(queries.extent(0));
  expanded_labels.reserve(queries.extent(0));

  for (uint32_t query = 0; query < static_cast<uint32_t>(normalized_queries.labels_by_query.size()); ++query) {
    for (auto label : normalized_queries.labels_by_query[query]) {
      expanded_query_ids.push_back(query);
      expanded_labels.push_back(label);
    }
  }
  if (expanded_query_ids.empty()) { return; }

  auto expanded_queries = gather_queries(res, queries, expanded_query_ids);
  auto d_expanded_query_ids = raft::make_device_vector<uint32_t, int64_t>(
    res, static_cast<int64_t>(expanded_query_ids.size()));
  auto d_expanded_labels = raft::make_device_vector<uint32_t, int64_t>(
    res, static_cast<int64_t>(expanded_labels.size()));
  auto stream = raft::resource::get_cuda_stream(res);
  raft::update_device(d_expanded_query_ids.data_handle(),
                      expanded_query_ids.data(),
                      static_cast<int64_t>(expanded_query_ids.size()),
                      stream);
  raft::update_device(d_expanded_labels.data_handle(),
                      expanded_labels.data(),
                      static_cast<int64_t>(expanded_labels.size()),
                      stream);

  auto topk = neighbors.extent(1);
  auto sub_neighbors = raft::make_device_matrix<uint32_t, int64_t>(
    res, static_cast<int64_t>(expanded_query_ids.size()), topk);
  auto sub_distances = raft::make_device_matrix<float, int64_t>(
    res, static_cast<int64_t>(expanded_query_ids.size()), topk);
  cuvs::neighbors::vecflow::detail::search<data_t>(res,
                 index,
                 raft::make_const_mdspan(expanded_queries.view()),
                 d_expanded_labels.view(),
                 itopk_size,
                 sub_neighbors.view(),
                 sub_distances.view());

  merge_device_results_to_output(res,
                                 d_expanded_query_ids.view(),
                                 sub_neighbors.view(),
                                 sub_distances.view(),
                                 queries.extent(0),
                                 static_cast<int>(topk),
                                 neighbors,
                                 distances,
                                 false);
}

template <typename data_t>
inline void search_multi_label_and(shared_resources::configured_raft_resources& res,
                                   cuvs::neighbors::vecflow::index<data_t>& index,
                                   raft::device_matrix_view<const data_t, int64_t> queries,
                                   const host_multi_label_query_desc& query_desc,
                                   int itopk_size,
                                   multi_label_query_desc::and_strategy strategy,
                                   raft::device_matrix_view<uint32_t, int64_t> neighbors,
                                   raft::device_matrix_view<float, int64_t> distances)
{
  auto const& host_cat_freq = copy_cat_freq_to_host(res, index);
  auto normalized_queries = normalize_query_labels_for_search(index, query_desc);
  auto query_count = static_cast<uint32_t>(normalized_queries.labels_by_query.size());
  if (query_count == 0) { return; }

  auto d_query_label_offsets = raft::make_device_vector<int64_t, int64_t>(
    res, static_cast<int64_t>(normalized_queries.label_offsets.size()));
  auto d_query_label_indices = raft::make_device_vector<uint32_t, int64_t>(
    res, static_cast<int64_t>(normalized_queries.label_indices.size()));
  auto stream = raft::resource::get_cuda_stream(res);
  if (!normalized_queries.label_offsets.empty()) {
    raft::update_device(d_query_label_offsets.data_handle(),
                        normalized_queries.label_offsets.data(),
                        static_cast<int64_t>(normalized_queries.label_offsets.size()),
                        stream);
  }
  if (!normalized_queries.label_indices.empty()) {
    raft::update_device(d_query_label_indices.data_handle(),
                        normalized_queries.label_indices.data(),
                        static_cast<int64_t>(normalized_queries.label_indices.size()),
                        stream);
  }

  auto topk = static_cast<int>(neighbors.extent(1));
  auto candidate_topk = std::max(topk, itopk_size);

  struct device_subquery_batch {
    raft::device_vector<uint32_t, int64_t> query_ids;
    raft::device_matrix<uint32_t, int64_t> neighbors;
    raft::device_matrix<float, int64_t> distances;
  };

  auto run_subquery_batch = [&](const std::vector<uint32_t>& batch_query_ids,
                                const std::vector<uint32_t>& batch_labels)
    -> std::optional<device_subquery_batch> {
    if (batch_query_ids.empty()) { return std::nullopt; }

    auto subqueries = gather_queries(res, queries, batch_query_ids);
    auto d_batch_query_ids = raft::make_device_vector<uint32_t, int64_t>(
      res, static_cast<int64_t>(batch_query_ids.size()));
    auto d_subquery_labels = raft::make_device_vector<uint32_t, int64_t>(
      res, static_cast<int64_t>(batch_labels.size()));
    raft::update_device(d_batch_query_ids.data_handle(),
                        batch_query_ids.data(),
                        static_cast<int64_t>(batch_query_ids.size()),
                        stream);
    raft::update_device(d_subquery_labels.data_handle(),
                        batch_labels.data(),
                        static_cast<int64_t>(batch_labels.size()),
                        stream);

    auto sub_neighbors = raft::make_device_matrix<uint32_t, int64_t>(
      res, static_cast<int64_t>(batch_query_ids.size()), candidate_topk);
    auto sub_distances = raft::make_device_matrix<float, int64_t>(
      res, static_cast<int64_t>(batch_query_ids.size()), candidate_topk);
    cuvs::neighbors::vecflow::detail::search<data_t>(res,
                   index,
                   raft::make_const_mdspan(subqueries.view()),
                   d_subquery_labels.view(),
                   candidate_topk,
                   sub_neighbors.view(),
                   sub_distances.view());

    return device_subquery_batch{std::move(d_batch_query_ids),
                                 std::move(sub_neighbors),
                                 std::move(sub_distances)};
  };

  auto merge_filtered_batch = [&](device_subquery_batch& batch, bool include_existing_output) {
    filter_device_results_by_query_membership(res,
                                              index,
                                              d_query_label_indices.view(),
                                              d_query_label_offsets.view(),
                                              batch.query_ids.view(),
                                              batch.neighbors.view(),
                                              batch.distances.view(),
                                              queries.extent(0));
    merge_device_results_to_output(res,
                                   batch.query_ids.view(),
                                   batch.neighbors.view(),
                                   batch.distances.view(),
                                   queries.extent(0),
                                   topk,
                                   neighbors,
                                   distances,
                                   include_existing_output);
  };

  if (strategy == multi_label_query_desc::and_strategy::PARALLEL) {
    std::vector<uint32_t> batch_query_ids;
    std::vector<uint32_t> batch_labels;
    for (uint32_t query = 0; query < query_count; ++query) {
      for (auto label : normalized_queries.labels_by_query[query]) {
        batch_query_ids.push_back(query);
        batch_labels.push_back(label);
      }
    }
    auto batch = run_subquery_batch(batch_query_ids, batch_labels);
    if (!batch.has_value()) { return; }
    merge_filtered_batch(*batch, false);
    return;
  }

  std::vector<uint32_t> initial_query_ids;
  std::vector<uint32_t> initial_labels;
  std::vector<std::vector<uint32_t>> fallback_labels_by_query(query_count);
  initial_query_ids.reserve(query_count);
  initial_labels.reserve(query_count);

  for (uint32_t query = 0; query < query_count; ++query) {
    auto const& labels = normalized_queries.labels_by_query[query];
    if (labels.empty()) { continue; }

    auto pivot = labels.front();
    auto pivot_freq =
      pivot < host_cat_freq.size() ? host_cat_freq[pivot] : std::numeric_limits<uint32_t>::max();
    for (auto label : labels) {
      auto freq = label < host_cat_freq.size() ? host_cat_freq[label]
                                               : std::numeric_limits<uint32_t>::max();
      if (freq < pivot_freq) {
        pivot = label;
        pivot_freq = freq;
      }
    }

    initial_query_ids.push_back(query);
    initial_labels.push_back(pivot);
    auto& fallback_labels = fallback_labels_by_query[query];
    for (auto label : labels) {
      if (label != pivot) { fallback_labels.push_back(label); }
    }
    std::sort(fallback_labels.begin(),
              fallback_labels.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                auto lhs_freq = lhs < host_cat_freq.size() ? host_cat_freq[lhs]
                                                           : std::numeric_limits<uint32_t>::max();
                auto rhs_freq = rhs < host_cat_freq.size() ? host_cat_freq[rhs]
                                                           : std::numeric_limits<uint32_t>::max();
                if (lhs_freq != rhs_freq) { return lhs_freq < rhs_freq; }
                return lhs < rhs;
              });
  }

  auto initial_batch = run_subquery_batch(initial_query_ids, initial_labels);
  if (!initial_batch.has_value()) { return; }
  merge_filtered_batch(*initial_batch, false);

  auto valid_counts = count_valid_results_in_output(res, neighbors);
  std::vector<std::size_t> next_fallback_index(query_count, 0);

  while (true) {
    std::vector<uint32_t> fallback_query_ids;
    std::vector<uint32_t> fallback_labels;
    fallback_query_ids.reserve(query_count);
    fallback_labels.reserve(query_count);

    for (uint32_t query = 0; query < query_count; ++query) {
      if (valid_counts[query] >= static_cast<uint32_t>(topk)) { continue; }
      auto& labels = fallback_labels_by_query[query];
      if (next_fallback_index[query] >= labels.size()) { continue; }
      fallback_query_ids.push_back(query);
      fallback_labels.push_back(labels[next_fallback_index[query]]);
      next_fallback_index[query] += 1;
    }

    if (fallback_query_ids.empty()) { break; }
    auto fallback_batch = run_subquery_batch(fallback_query_ids, fallback_labels);
    if (!fallback_batch.has_value()) { break; }
    merge_filtered_batch(*fallback_batch, true);
    valid_counts = count_valid_results_in_output(res, neighbors);
  }
}

} // namespace detail

template<typename data_t>
void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<data_t>& index,
            raft::device_matrix_view<const data_t, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances)
{
  cuvs::neighbors::vecflow::detail::search<data_t>(
    res, index, queries, query_labels, itopk_size, neighbors, distances);
}

template<typename data_t>
void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<data_t>& index,
            raft::device_matrix_view<const data_t, int64_t> queries,
            const multi_label_query_desc& query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances)
{
  initialize_search_results(res, neighbors, distances);
  auto host_query_desc = detail::copy_multi_label_query_desc_to_host(res, query_labels);
  if (host_query_desc.label_offsets.size() != static_cast<std::size_t>(queries.extent(0) + 1)) {
    throw std::invalid_argument("Multi-label query offsets must have length n_queries + 1");
  }

  if (query_labels.mode == multi_label_query_desc::combine_mode::OR) {
    detail::search_multi_label_or(
      res, index, queries, host_query_desc, itopk_size, neighbors, distances);
  } else {
    detail::search_multi_label_and(res,
                                   index,
                                   queries,
                                   host_query_desc,
                                   itopk_size,
                                   query_labels.and_mode,
                                   neighbors,
                                   distances);
  }
}

inline void search_multi_gpu_impl(shared_resources::configured_raft_resources& res,
                                  cuvs::neighbors::vecflow::multi_gpu_index<float>& index,
                                  raft::device_matrix_view<const float, int64_t> queries,
                                  raft::device_vector_view<uint32_t, int64_t> query_labels,
                                  int itopk_size,
                                  raft::device_matrix_view<uint32_t, int64_t> neighbors,
                                  raft::device_matrix_view<float, int64_t> distances)
{
  if (index.workers.empty()) {
    throw std::invalid_argument("search_multi_gpu requires at least one worker");
  }
  if (neighbors.extent(0) != queries.extent(0) || distances.extent(0) != queries.extent(0) ||
      neighbors.extent(1) != distances.extent(1)) {
    throw std::invalid_argument("search_multi_gpu output shapes do not match the query count");
  }

  auto host_queries = detail::multi_gpu::copy_device_matrix_to_host(res, queries);
  std::vector<uint32_t> host_query_labels(static_cast<std::size_t>(queries.extent(0)));
  raft::copy(host_query_labels.data(),
             query_labels.data_handle(),
             queries.extent(0),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  auto worker_query_ids = detail::multi_gpu::route_queries_to_workers(
    host_query_labels, index.label_to_worker, index.workers.size());

  auto topk = neighbors.extent(1);
  std::vector<uint32_t> host_neighbors(static_cast<std::size_t>(queries.extent(0) * topk),
                                       UINT32_MAX);
  std::vector<float> host_distances(static_cast<std::size_t>(queries.extent(0) * topk),
                                    std::numeric_limits<float>::infinity());
  std::vector<std::string> worker_errors;
  std::mutex worker_error_mutex;
  std::vector<std::thread> workers;
  workers.reserve(index.workers.size());

  for (std::size_t worker_idx = 0; worker_idx < index.workers.size(); ++worker_idx) {
    if (worker_query_ids[worker_idx].empty()) { continue; }
    workers.emplace_back([&, worker_idx]() {
      try {
        auto& worker = index.workers[worker_idx];
        if (!worker.resources || !worker.index.has_value() || !worker.dataset.has_value()) {
          throw std::runtime_error("search_multi_gpu worker is missing a built index or dataset");
        }

        RAFT_CUDA_TRY(cudaSetDevice(worker.device_id));
        shared_resources::thread_id = static_cast<int>(worker_idx);
        shared_resources::n_threads = static_cast<int>(index.workers.size());
        shared_resources::configured_raft_resources worker_res(*worker.resources);

        std::vector<float> subset_queries;
        std::vector<uint32_t> subset_labels;
        detail::multi_gpu::gather_host_query_subset(host_queries,
                                                    queries.extent(1),
                                                    host_query_labels,
                                                    worker_query_ids[worker_idx],
                                                    &subset_queries,
                                                    &subset_labels);

        auto query_count = static_cast<int64_t>(worker_query_ids[worker_idx].size());
        auto d_queries =
          raft::make_device_matrix<float, int64_t>(worker_res, query_count, queries.extent(1));
        auto d_query_labels = raft::make_device_vector<uint32_t, int64_t>(worker_res, query_count);
        raft::copy(d_queries.data_handle(),
                   subset_queries.data(),
                   query_count * queries.extent(1),
                   raft::resource::get_cuda_stream(worker_res));
        raft::copy(d_query_labels.data_handle(),
                   subset_labels.data(),
                   query_count,
                   raft::resource::get_cuda_stream(worker_res));

        auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(worker_res, query_count, topk);
        auto d_distances = raft::make_device_matrix<float, int64_t>(worker_res, query_count, topk);
        cuvs::neighbors::vecflow::search(worker_res,
                                         *worker.index,
                                         raft::make_const_mdspan(d_queries.view()),
                                         d_query_labels.view(),
                                         itopk_size,
                                         d_neighbors.view(),
                                         d_distances.view());
        raft::resource::sync_stream(worker_res);

        std::vector<uint32_t> subset_neighbors(static_cast<std::size_t>(query_count * topk));
        std::vector<float> subset_distances(static_cast<std::size_t>(query_count * topk));
        raft::copy(subset_neighbors.data(),
                   d_neighbors.data_handle(),
                   query_count * topk,
                   raft::resource::get_cuda_stream(worker_res));
        raft::copy(subset_distances.data(),
                   d_distances.data_handle(),
                   query_count * topk,
                   raft::resource::get_cuda_stream(worker_res));
        raft::resource::sync_stream(worker_res);

        for (std::size_t local_idx = 0; local_idx < worker_query_ids[worker_idx].size(); ++local_idx) {
          auto global_query = worker_query_ids[worker_idx][local_idx];
          auto host_offset  = global_query * topk;
          auto subset_offset = static_cast<int64_t>(local_idx) * topk;
          std::copy_n(
            subset_neighbors.data() + subset_offset, topk, host_neighbors.data() + host_offset);
          std::copy_n(
            subset_distances.data() + subset_offset, topk, host_distances.data() + host_offset);
        }
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

  for (auto& worker : workers) { worker.join(); }
  if (!worker_errors.empty()) {
    std::string message = "search_multi_gpu worker failures:";
    for (auto const& error : worker_errors) {
      message += "\n";
      message += error;
    }
    throw std::runtime_error(message);
  }

  raft::copy(neighbors.data_handle(),
             host_neighbors.data(),
             queries.extent(0) * topk,
             raft::resource::get_cuda_stream(res));
  raft::copy(distances.data_handle(),
             host_distances.data(),
             queries.extent(0) * topk,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
}

} // namespace cuvs::neighbors::vecflow
