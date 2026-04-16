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
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <future>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
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

template <typename data_t>
void cache_label_graph_in_hbm(cuvs::neighbors::vecflow::index<data_t>& index,
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

  cache_label_graph_in_hbm(index, label, label_offset, label_size, requested_bytes, graph_storage);
  return graph_storage;
}

template <typename data_t>
void cache_label_graph_in_hbm(cuvs::neighbors::vecflow::index<data_t>& index,
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
    auto evict_label =
      lowest_score_cached_label_locked(index, index.phoenix_label_cache->graphs);
    if (evict_label == std::numeric_limits<uint32_t>::max()) { break; }
    auto evict_it = index.phoenix_label_cache->graphs.find(evict_label);
    if (evict_it == index.phoenix_label_cache->graphs.end()) { continue; }
    vecflow_log("Evicting HBM graph cache for label ", evict_label, " using score-based policy");
    index.phoenix_label_cache->lru_labels.erase(evict_it->second.lru_it);
    index.phoenix_label_cache->cached_bytes -= evict_it->second.bytes;
    index.phoenix_label_cache->graphs.erase(evict_it);
  }

  if (index.phoenix_label_cache->cached_bytes + requested_bytes > cache_capacity) { return; }

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
  auto cache_capacity = index.phoenix_label_dram_cache_capacity_bytes;
  if (index.phoenix_label_cache == nullptr || cache_capacity == 0 || requested_bytes > cache_capacity) {
    return;
  }

  std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
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
    auto evict_label =
      lowest_score_cached_label_locked(index, index.phoenix_label_cache->host_graphs);
    if (evict_label == std::numeric_limits<uint32_t>::max()) { break; }
    auto evict_it = index.phoenix_label_cache->host_graphs.find(evict_label);
    if (evict_it == index.phoenix_label_cache->host_graphs.end()) { continue; }
    vecflow_log("Evicting DRAM graph cache for label ", evict_label, " using score-based policy");
    index.phoenix_label_cache->host_lru_labels.erase(evict_it->second.lru_it);
    index.phoenix_label_cache->host_cached_bytes -= evict_it->second.bytes;
    index.phoenix_label_cache->host_graphs.erase(evict_it);
  }

  if (index.phoenix_label_cache->host_cached_bytes + requested_bytes > cache_capacity) { return; }

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
void cache_label_dataset_in_hbm(cuvs::neighbors::vecflow::index<data_t>& index,
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
    index, label, label_offset, label_size, requested_bytes, dataset_storage);
  return dataset_storage;
}

template <typename data_t>
void cache_label_dataset_in_hbm(cuvs::neighbors::vecflow::index<data_t>& index,
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
    auto evict_label =
      lowest_score_cached_label_locked(index, index.phoenix_label_dataset_cache->datasets);
    if (evict_label == std::numeric_limits<uint32_t>::max()) { break; }
    auto evict_it = index.phoenix_label_dataset_cache->datasets.find(evict_label);
    if (evict_it == index.phoenix_label_dataset_cache->datasets.end()) { continue; }
    vecflow_log("Evicting HBM dataset cache for label ", evict_label, " using score-based policy");
    index.phoenix_label_dataset_cache->lru_labels.erase(evict_it->second.lru_it);
    index.phoenix_label_dataset_cache->cached_bytes -= evict_it->second.bytes;
    index.phoenix_label_dataset_cache->datasets.erase(evict_it);
  }

  if (index.phoenix_label_dataset_cache->cached_bytes + requested_bytes > cache_capacity) { return; }

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
  auto cache_capacity = index.phoenix_label_dataset_dram_cache_capacity_bytes;
  if (index.phoenix_label_dataset_cache == nullptr || cache_capacity == 0 ||
      requested_bytes > cache_capacity) {
    return;
  }

  std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
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
    auto evict_label =
      lowest_score_cached_label_locked(index, index.phoenix_label_dataset_cache->host_datasets);
    if (evict_label == std::numeric_limits<uint32_t>::max()) { break; }
    auto evict_it = index.phoenix_label_dataset_cache->host_datasets.find(evict_label);
    if (evict_it == index.phoenix_label_dataset_cache->host_datasets.end()) { continue; }
    vecflow_log("Evicting DRAM dataset cache for label ", evict_label, " using score-based policy");
    index.phoenix_label_dataset_cache->host_lru_labels.erase(evict_it->second.lru_it);
    index.phoenix_label_dataset_cache->host_cached_bytes -= evict_it->second.bytes;
    index.phoenix_label_dataset_cache->host_datasets.erase(evict_it);
  }

  if (index.phoenix_label_dataset_cache->host_cached_bytes + requested_bytes > cache_capacity) {
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

  {
    std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
    for (auto const& [label, entry] : index.phoenix_label_cache->graphs) {
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
      best_dram_score <= worst_hbm_score || promote_label >= index.host_cagra_label_size.size()) {
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
    index, promote_label, promote_label_offset, promote_label_size, promote_bytes, device_graph_storage);
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

  {
    std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
    for (auto const& [label, entry] : index.phoenix_label_dataset_cache->datasets) {
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
      best_dram_score <= worst_hbm_score || promote_label >= index.host_cagra_label_size.size()) {
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
  cache_label_dataset_in_hbm(index,
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
      cache_label_graph_in_hbm(
        index, label, label_offset, label_size, requested_bytes, device_graph_storage);
      return device_graph_storage;
    }
  }

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
      cache_label_dataset_in_hbm(
        index, label, label_offset, label_size, requested_bytes, device_dataset_storage);
      return device_dataset_storage;
    }
  }

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
  int topk)
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

  auto stream          = raft::resource::get_cuda_stream(res);
  auto query_dim       = query_info.cagra_queries.extent(1);
  auto total_graph_rows = static_cast<int64_t>(index.cagra_index_map.size());
  auto num_cagra_queries = static_cast<int64_t>(query_info.cagra_query_map.size());

  std::vector<uint32_t> host_query_labels(num_cagra_queries);
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

  int64_t max_group_size = 0;
  int64_t max_label_size = 0;
  for (auto label : label_order) {
    max_group_size =
      std::max(max_group_size, static_cast<int64_t>(query_positions_by_label.at(label).size()));
    max_label_size =
      std::max(max_label_size, static_cast<int64_t>(index.host_cagra_label_size.at(label)));
  }
  if (max_group_size == 0 || max_label_size == 0) { return; }

  auto scratch_positions = raft::make_device_vector<uint32_t, int64_t>(res, max_group_size);
  auto scratch_queries   = raft::make_device_matrix<data_t, int64_t>(res, max_group_size, query_dim);
  auto scratch_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, max_group_size);
  auto scratch_index_map = raft::make_device_vector<uint32_t, int64_t>(res, max_label_size);
  auto scratch_label_size = raft::make_device_vector<uint32_t, int64_t>(res, 1);
  auto scratch_label_offset = raft::make_device_vector<uint32_t, int64_t>(res, 1);
  auto scratch_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, max_group_size, topk);
  auto scratch_distances = raft::make_device_matrix<float, int64_t>(res, max_group_size, topk);
  uint32_t host_group_label_offset = 0;
  raft::update_device(
    scratch_label_offset.data_handle(), &host_group_label_offset, 1, stream);

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

    for (auto future_index = current_label_position + 1; future_index < label_order.size();
         ++future_index) {
      auto future_label = label_order[future_index];
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

  for (std::size_t label_position = 0; label_position < label_order.size(); ++label_position) {
    auto label = label_order[label_position];
    if (label >= index.host_cagra_label_size.size()) {
      throw std::runtime_error("Phoenix label-load query label is out of bounds: " +
                               std::to_string(label));
    }

    auto label_size   = static_cast<int64_t>(index.host_cagra_label_size[label]);
    auto label_offset = static_cast<int64_t>(index.host_cagra_label_offset[label]);
    if (label_size <= 0) {
      throw std::runtime_error("Phoenix label-load received a CAGRA query for non-CAGRA label " +
                               std::to_string(label));
    }
    record_phoenix_label_access(res, index, label, query_dim);

    auto const& host_positions = query_positions_by_label.at(label);
    auto group_size            = static_cast<int64_t>(host_positions.size());
    raft::update_device(
      scratch_positions.data_handle(), host_positions.data(), group_size, stream);

    auto total_values = group_size * query_dim;
    auto block_size   = 256;
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

    uint32_t host_group_label_size   = static_cast<uint32_t>(label_size);
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
                                               label_order,
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
      group_label_offset_view);

    auto total_results = group_size * topk;
    auto result_grid   = static_cast<int>((total_results + block_size - 1) / block_size);
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
            raft::device_matrix_view<float, int64_t> distances) {

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
      search_cagra_with_phoenix_label_load(
        res, index, search_params, query_info, cagra_neighbors.view(), cagra_distances.view(), topk);
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
                             index.cagra_label_offset.view());
    }
    raft::resource::sync_stream(res);
  }

  if (n_bfs_queries > 0) {
    search_filtered_bfs(res,
												index.ivf_bfs_index,
												raft::make_const_mdspan(query_info.bfs_queries.view()),
												query_info.bfs_query_labels.view(),
												index.bfs_label_size.view(),
												bfs_neighbors.view(),
												bfs_distances.view(),
												cuvs::distance::DistanceType::L2Unexpanded);
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
