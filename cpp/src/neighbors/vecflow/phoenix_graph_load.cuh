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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
#include "phoenix.h"
#endif

namespace cuvs::neighbors::vecflow::detail::phoenix {

inline constexpr std::size_t kPhoenixPageAlignment = 64 * 1024;
inline constexpr off_t kIbinHeaderBytes            = static_cast<off_t>(sizeof(std::int64_t) * 2);

inline auto env_truthy(const char* value) -> bool
{
  if (value == nullptr) { return false; }
  return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "ON") == 0 || std::strcmp(value, "yes") == 0 ||
         std::strcmp(value, "YES") == 0;
}

inline auto env_uint64_or_default(const char* value, std::uint64_t default_value) -> std::uint64_t
{
  if (value == nullptr || *value == '\0') { return default_value; }
  char* end = nullptr;
  auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0') { return default_value; }
  return static_cast<std::uint64_t>(parsed);
}

inline auto use_phoenix_graph_load() -> bool
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return env_truthy(std::getenv("CUVS_VECFLOW_USE_PHOENIX_GRAPH_LOAD"));
#else
  return false;
#endif
}

inline auto use_phoenix_label_load() -> bool
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return env_truthy(std::getenv("CUVS_VECFLOW_USE_PHOENIX_LABEL_LOAD"));
#else
  return false;
#endif
}

inline auto phoenix_label_cache_bytes() -> std::size_t
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return static_cast<std::size_t>(env_uint64_or_default(
    std::getenv("CUVS_VECFLOW_PHOENIX_LABEL_CACHE_BYTES"), 1ULL << 30));
#else
  return 0;
#endif
}

inline auto phoenix_label_dram_cache_bytes() -> std::size_t
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return static_cast<std::size_t>(env_uint64_or_default(
    std::getenv("CUVS_VECFLOW_PHOENIX_LABEL_DRAM_CACHE_BYTES"), 0));
#else
  return 0;
#endif
}

inline auto phoenix_label_prefetch_max_bytes() -> std::size_t
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return static_cast<std::size_t>(env_uint64_or_default(
    std::getenv("CUVS_VECFLOW_PHOENIX_LABEL_PREFETCH_MAX_BYTES"), 0));
#else
  return 0;
#endif
}

inline auto phoenix_label_rebalance_interval_queries() -> std::size_t
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return static_cast<std::size_t>(env_uint64_or_default(
    std::getenv("CUVS_VECFLOW_PHOENIX_LABEL_REBALANCE_INTERVAL_QUERIES"), 64));
#else
  return 0;
#endif
}

inline auto phoenix_label_dataset_cache_bytes() -> std::size_t
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return static_cast<std::size_t>(env_uint64_or_default(
    std::getenv("CUVS_VECFLOW_PHOENIX_LABEL_DATASET_CACHE_BYTES"),
    phoenix_label_cache_bytes()));
#else
  return 0;
#endif
}

inline auto phoenix_label_dataset_dram_cache_bytes() -> std::size_t
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return static_cast<std::size_t>(env_uint64_or_default(
    std::getenv("CUVS_VECFLOW_PHOENIX_LABEL_DATASET_DRAM_CACHE_BYTES"),
    phoenix_label_dram_cache_bytes()));
#else
  return 0;
#endif
}

inline auto phoenix_label_dataset_prefetch_max_bytes() -> std::size_t
{
#ifdef CUVS_VECFLOW_PHOENIX_ENABLED
  return static_cast<std::size_t>(env_uint64_or_default(
    std::getenv("CUVS_VECFLOW_PHOENIX_LABEL_DATASET_PREFETCH_MAX_BYTES"),
    phoenix_label_prefetch_max_bytes()));
#else
  return 0;
#endif
}

#ifdef CUVS_VECFLOW_PHOENIX_ENABLED

inline auto round_up(std::size_t value, std::size_t alignment) -> std::size_t
{
  return ((value + alignment - 1) / alignment) * alignment;
}

inline auto verbose_logging() -> bool
{
  return env_truthy(std::getenv("CUVS_VECFLOW_VERBOSE"));
}

template <typename... Args>
inline void phoenix_log(Args&&... args)
{
  if (!verbose_logging()) { return; }
  ((std::cout << std::forward<Args>(args)), ...);
  std::cout << std::endl;
}

struct ibin_metadata {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
};

inline auto read_ibin_metadata(const std::string& filename) -> ibin_metadata
{
  thread_local std::unordered_map<std::string, ibin_metadata> metadata_cache;
  auto it = metadata_cache.find(filename);
  if (it != metadata_cache.end()) { return it->second; }

  std::ifstream metadata_file(filename, std::ios::binary);
  if (!metadata_file) { throw std::runtime_error("Cannot open file: " + filename); }

  ibin_metadata metadata;
  metadata_file.read(reinterpret_cast<char*>(&metadata.rows), sizeof(metadata.rows));
  metadata_file.read(reinterpret_cast<char*>(&metadata.cols), sizeof(metadata.cols));
  if (!metadata_file) { throw std::runtime_error("Cannot read ibin header from: " + filename); }
  metadata_cache.emplace(filename, metadata);
  return metadata;
}

inline auto resolve_phoenix_device_id() -> int
{
  if (auto* visible = std::getenv("CUDA_VISIBLE_DEVICES"); visible != nullptr &&
                                                           std::strchr(visible, ',') == nullptr) {
    char* end = nullptr;
    auto value = std::strtol(visible, &end, 10);
    if (end != visible && *end == '\0' && value >= 0) { return static_cast<int>(value); }
  }

  int current_device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&current_device));
  return current_device;
}

template <typename T>
inline void free_device_buffer(int cuda_device_ordinal, T* storage)
{
  if (storage == nullptr) { return; }
  int original_device = 0;
  auto get_device_ret = cudaGetDevice(&original_device);
  bool restore_device = get_device_ret == cudaSuccess && original_device != cuda_device_ordinal;
  if (restore_device) { cudaSetDevice(cuda_device_ordinal); }
  auto free_ret = cudaFree(storage);
  if (restore_device) { cudaSetDevice(original_device); }
  if (free_ret != cudaSuccess) {
    std::fprintf(stderr,
                 "Warning: cudaFree failed for Phoenix graph buffer on CUDA device %d: %s\n",
                 cuda_device_ordinal,
                 cudaGetErrorString(free_ret));
  }
}

template <typename T>
class phoenix_buffer_slot {
 public:
  phoenix_buffer_slot(int cuda_device_ordinal, int phoenix_device_id)
    : cuda_device_ordinal_(cuda_device_ordinal),
      phoenix_device_id_(phoenix_device_id)
  {
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    stream_created_ = true;
  }

  phoenix_buffer_slot(const phoenix_buffer_slot&) = delete;
  auto operator=(const phoenix_buffer_slot&) -> phoenix_buffer_slot& = delete;
  phoenix_buffer_slot(phoenix_buffer_slot&&) = delete;
  auto operator=(phoenix_buffer_slot&&) -> phoenix_buffer_slot& = delete;

  ~phoenix_buffer_slot() { cleanup(); }

  [[nodiscard]] auto data() const -> T* { return storage_; }
  [[nodiscard]] auto stream() const -> cudaStream_t { return stream_; }
  [[nodiscard]] auto capacity_bytes() const -> std::size_t { return capacity_bytes_; }

  auto try_acquire() -> bool
  {
    auto expected = false;
    return in_use_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
  }

  void release() { in_use_.store(false, std::memory_order_release); }

  void ensure_capacity(std::size_t requested_bytes)
  {
    if (capacity_bytes_ >= requested_bytes && storage_ != nullptr && registered_) { return; }

    if (stream_ != nullptr) {
      auto sync_ret = cudaStreamSynchronize(stream_);
      if (sync_ret != cudaSuccess) {
        throw std::runtime_error("cudaStreamSynchronize failed while resizing Phoenix slot: " +
                                 std::string(cudaGetErrorString(sync_ret)));
      }
    }
    deregister_buffer();
    if (storage_ != nullptr) {
      free_device_buffer(cuda_device_ordinal_, storage_);
      storage_ = nullptr;
    }

    RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&storage_), requested_bytes));
    void* target_addr = nullptr;
    if (phxfs_regmem(phoenix_device_id_, storage_, requested_bytes, &target_addr) != 0) {
      free_device_buffer(cuda_device_ordinal_, storage_);
      storage_ = nullptr;
      throw std::runtime_error("phxfs_regmem failed while allocating Phoenix slot");
    }
    registered_       = true;
    registered_bytes_ = requested_bytes;
    capacity_bytes_   = requested_bytes;
  }

 private:
  void deregister_buffer() noexcept
  {
    if (!registered_ || storage_ == nullptr) { return; }
    auto ret = phxfs_deregmem(phoenix_device_id_, storage_, registered_bytes_);
    if (ret != 0) {
      std::fprintf(stderr,
                   "Warning: phxfs_deregmem failed for Phoenix device %d with code %d\n",
                   phoenix_device_id_,
                   ret);
    }
    registered_       = false;
    registered_bytes_ = 0;
  }

  void cleanup() noexcept
  {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
    }
    deregister_buffer();
    if (storage_ != nullptr) {
      free_device_buffer(cuda_device_ordinal_, storage_);
      storage_ = nullptr;
    }
    if (stream_created_ && stream_ != nullptr) {
      auto destroy_ret = cudaStreamDestroy(stream_);
      if (destroy_ret != cudaSuccess) {
        std::fprintf(stderr,
                     "Warning: cudaStreamDestroy failed for Phoenix prefetch stream: %s\n",
                     cudaGetErrorString(destroy_ret));
      }
      stream_         = nullptr;
      stream_created_ = false;
    }
  }

  int cuda_device_ordinal_ = 0;
  int phoenix_device_id_   = 0;
  T* storage_              = nullptr;
  std::size_t capacity_bytes_   = 0;
  std::size_t registered_bytes_ = 0;
  bool registered_              = false;
  cudaStream_t stream_          = nullptr;
  bool stream_created_          = false;
  std::atomic<bool> in_use_{false};
};

template <typename T>
class phoenix_file_session : public std::enable_shared_from_this<phoenix_file_session<T>> {
 public:
  using slot_type = phoenix_buffer_slot<T>;

  phoenix_file_session(const std::string& filename,
                       std::int64_t expected_rows,
                       std::int64_t expected_cols)
    : filename_(filename), metadata_(read_ibin_metadata(filename))
  {
    if (metadata_.rows != expected_rows || metadata_.cols != expected_cols) {
      throw std::runtime_error("File dimensions do not match cached VecFlow graph dimensions");
    }

    fd_ = ::open(filename.c_str(), O_RDONLY);
    if (fd_ < 0) { throw std::runtime_error("Cannot open file descriptor for: " + filename); }

    RAFT_CUDA_TRY(cudaGetDevice(&cuda_device_ordinal_));
    phoenix_device_id_ = resolve_phoenix_device_id();
    if (phxfs_open(phoenix_device_id_) != 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("phxfs_open failed for Phoenix device " +
                               std::to_string(phoenix_device_id_));
    }
    opened_ = true;
  }

  phoenix_file_session(const phoenix_file_session&) = delete;
  auto operator=(const phoenix_file_session&) -> phoenix_file_session& = delete;
  phoenix_file_session(phoenix_file_session&&) = delete;
  auto operator=(phoenix_file_session&&) -> phoenix_file_session& = delete;

  ~phoenix_file_session() { cleanup(); }

  [[nodiscard]] auto rows() const -> std::int64_t { return metadata_.rows; }
  [[nodiscard]] auto cols() const -> std::int64_t { return metadata_.cols; }

  [[nodiscard]] auto file_id() const -> phxfs_fileid_t
  {
    phxfs_fileid_t file_id{};
    file_id.fd       = fd_;
    file_id.deviceID = phoenix_device_id_;
    return file_id;
  }

  auto acquire_slot(std::size_t requested_bytes) -> std::shared_ptr<slot_type>
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto const& slot : slots_) {
      if (!slot->try_acquire()) { continue; }
      try {
        slot->ensure_capacity(requested_bytes);
        return slot;
      } catch (...) {
        slot->release();
        throw;
      }
    }

    auto slot = std::make_shared<slot_type>(cuda_device_ordinal_, phoenix_device_id_);
    if (!slot->try_acquire()) {
      throw std::runtime_error("New Phoenix slot unexpectedly failed to acquire");
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

  auto release_slot(std::shared_ptr<slot_type> slot) -> std::shared_ptr<T>
  {
    auto self    = this->shared_from_this();
    auto* buffer = slot->data();
    return std::shared_ptr<T>(buffer, [self = std::move(self), slot = std::move(slot)](T*) mutable {
      slot->release();
    });
  }

 private:
  void cleanup() noexcept
  {
    slots_.clear();
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (opened_) {
      auto ret = phxfs_close(phoenix_device_id_);
      if (ret != 0) {
        std::fprintf(stderr,
                     "Warning: phxfs_close failed for Phoenix device %d with code %d\n",
                     phoenix_device_id_,
                     ret);
      }
      opened_ = false;
    }
  }

  std::string filename_;
  ibin_metadata metadata_;
  int cuda_device_ordinal_ = 0;
  int phoenix_device_id_   = 0;
  int fd_                  = -1;
  bool opened_             = false;
  std::mutex mutex_;
  std::vector<std::shared_ptr<slot_type>> slots_;
};

template <typename T>
inline auto get_file_session(const std::string& filename,
                             std::int64_t expected_rows,
                             std::int64_t expected_cols)
  -> std::shared_ptr<phoenix_file_session<T>>
{
  int cuda_device_ordinal = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&cuda_device_ordinal));
  thread_local std::unordered_map<std::string, std::weak_ptr<phoenix_file_session<T>>> sessions;
  auto cache_key = filename + "#" + std::to_string(cuda_device_ordinal);
  auto it = sessions.find(cache_key);
  if (it != sessions.end()) {
    if (auto existing = it->second.lock()) { return existing; }
  }

  auto session =
    std::make_shared<phoenix_file_session<T>>(filename, expected_rows, expected_cols);
  sessions[cache_key] = session;
  return session;
}

template <typename T>
class async_ibin_rows_request {
 public:
  async_ibin_rows_request(const std::string& filename,
                          std::int64_t total_rows,
                          std::int64_t expected_cols,
                          std::int64_t row_offset,
                          std::int64_t rows_to_load)
    : filename_(filename),
      row_offset_(row_offset),
      rows_to_load_(rows_to_load)
  {
    if (row_offset < 0 || rows_to_load <= 0 || row_offset + rows_to_load > total_rows) {
      throw std::invalid_argument("Requested Phoenix async graph row range is out of bounds");
    }

    session_ = get_file_session<T>(filename, total_rows, expected_cols);
    auto cols = session_->cols();

    try {
      data_bytes_ = static_cast<std::size_t>(rows_to_load) * static_cast<std::size_t>(cols) *
                    sizeof(T);
      aligned_bytes_ = round_up(data_bytes_, kPhoenixPageAlignment);
      file_offset_   = kIbinHeaderBytes + static_cast<off_t>(
                        row_offset * cols * static_cast<std::int64_t>(sizeof(T)));

      slot_ = session_->acquire_slot(aligned_bytes_);
      auto file_id = session_->file_id();
      bytes_done_  = 0;

      auto async_ret =
        phxfs_read_async(file_id, slot_->data(), data_bytes_, file_offset_, &bytes_done_,
                         reinterpret_cast<CUstream>(slot_->stream()));
      if (async_ret != cudaSuccess) {
        throw std::runtime_error("phxfs_read_async failed for file: " + filename + ": " +
                                 std::string(cudaGetErrorString(async_ret)));
      }

      started_ = true;
      phoenix_log("Prefetching rows [",
                  row_offset_,
                  ", ",
                  (row_offset_ + rows_to_load_),
                  ") from ",
                  filename_,
                  " through Phoenix");
    } catch (...) {
      cleanup();
      throw;
    }
  }

  async_ibin_rows_request(const async_ibin_rows_request&) = delete;
  auto operator=(const async_ibin_rows_request&) -> async_ibin_rows_request& = delete;
  async_ibin_rows_request(async_ibin_rows_request&&) = delete;
  auto operator=(async_ibin_rows_request&&) -> async_ibin_rows_request& = delete;

  ~async_ibin_rows_request() { cleanup(); }

  [[nodiscard]] auto row_offset() const -> std::int64_t { return row_offset_; }
  [[nodiscard]] auto rows_to_load() const -> std::int64_t { return rows_to_load_; }

  auto wait_and_release() -> std::shared_ptr<T>
  {
    sync();
    return session_->release_slot(std::move(slot_));
  }

 private:
  void sync()
  {
    if (!started_) { return; }
    RAFT_CUDA_TRY(cudaStreamSynchronize(slot_->stream()));
    if (bytes_done_ != static_cast<ssize_t>(data_bytes_)) {
      throw std::runtime_error("phxfs_read_async completed " + std::to_string(bytes_done_) +
                               " bytes from " + filename_ + ", expected " +
                               std::to_string(data_bytes_));
    }
    started_ = false;
  }

  void cleanup() noexcept
  {
    try {
      if (started_ && slot_ != nullptr) { cudaStreamSynchronize(slot_->stream()); }
    } catch (...) {
    }
    if (slot_ != nullptr) {
      slot_->release();
      slot_.reset();
    }
    started_ = false;
  }

  std::string filename_;
  std::int64_t row_offset_ = 0;
  std::int64_t rows_to_load_ = 0;
  std::size_t data_bytes_ = 0;
  std::size_t aligned_bytes_ = 0;
  off_t file_offset_ = 0;
  ssize_t bytes_done_ = 0;
  bool started_ = false;
  std::shared_ptr<phoenix_file_session<T>> session_;
  std::shared_ptr<phoenix_buffer_slot<T>> slot_;
};

using async_ibin_graph_rows_request = async_ibin_rows_request<uint32_t>;

template <typename T>
inline auto load_ibin_to_device(raft::resources const& res,
                                const std::string& filename,
                                std::int64_t expected_rows,
                                std::int64_t expected_cols)
  -> std::shared_ptr<T>
{
  (void)res;
  auto session    = get_file_session<T>(filename, expected_rows, expected_cols);
  auto data_bytes = static_cast<std::size_t>(session->rows()) * static_cast<std::size_t>(session->cols()) *
                    sizeof(T);
  auto aligned_bytes = round_up(data_bytes, kPhoenixPageAlignment);
  auto slot = session->acquire_slot(aligned_bytes);
  auto file_id = session->file_id();
  auto bytes_read = phxfs_read(file_id, slot->data(), 0, data_bytes, kIbinHeaderBytes);
  if (bytes_read != static_cast<ssize_t>(data_bytes)) {
    slot->release();
    throw std::runtime_error("phxfs_read read " + std::to_string(bytes_read) + " bytes from " +
                             filename + ", expected " + std::to_string(data_bytes));
  }

  phoenix_log("Loading matrix from ", filename, " through Phoenix");
  return session->release_slot(std::move(slot));
}

inline auto load_ibin_graph_to_device(raft::resources const& res,
                                      const std::string& filename,
                                      std::int64_t expected_rows,
                                      std::int64_t expected_cols)
  -> std::shared_ptr<uint32_t>
{
  return load_ibin_to_device<uint32_t>(res, filename, expected_rows, expected_cols);
}

template <typename T>
inline auto load_ibin_rows_to_device(raft::resources const& res,
                                     const std::string& filename,
                                     std::int64_t total_rows,
                                     std::int64_t expected_cols,
                                     std::int64_t row_offset,
                                     std::int64_t rows_to_load)
  -> std::shared_ptr<T>
{
  if (row_offset < 0 || rows_to_load <= 0 || row_offset + rows_to_load > total_rows) {
    throw std::invalid_argument("Requested Phoenix ibin row range is out of bounds");
  }
  (void)res;
  auto session = get_file_session<T>(filename, total_rows, expected_cols);
  auto data_bytes = static_cast<std::size_t>(rows_to_load) * static_cast<std::size_t>(session->cols()) *
                    sizeof(T);
  auto aligned_bytes = round_up(data_bytes, kPhoenixPageAlignment);
  auto file_offset =
    kIbinHeaderBytes +
    static_cast<off_t>(row_offset * session->cols() * static_cast<std::int64_t>(sizeof(T)));
  auto slot    = session->acquire_slot(aligned_bytes);
  auto file_id = session->file_id();
  auto bytes_read = phxfs_read(file_id, slot->data(), 0, data_bytes, file_offset);
  if (bytes_read != static_cast<ssize_t>(data_bytes)) {
    slot->release();
    throw std::runtime_error("phxfs_read read " + std::to_string(bytes_read) + " bytes from " +
                             filename + ", expected " + std::to_string(data_bytes));
  }

  phoenix_log("Loading rows [",
              row_offset,
              ", ",
              (row_offset + rows_to_load),
              ") from ",
              filename,
              " through Phoenix");
  return session->release_slot(std::move(slot));
}

inline auto load_ibin_graph_rows_to_device(raft::resources const& res,
                                           const std::string& filename,
                                           std::int64_t total_rows,
                                           std::int64_t expected_cols,
                                           std::int64_t row_offset,
                                           std::int64_t rows_to_load)
  -> std::shared_ptr<uint32_t>
{
  return load_ibin_rows_to_device<uint32_t>(
    res, filename, total_rows, expected_cols, row_offset, rows_to_load);
}

#endif

}  // namespace cuvs::neighbors::vecflow::detail::phoenix
