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

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuvs::neighbors::vecflow {

inline auto append_suffix_to_filename(const std::string& filename, const std::string& suffix)
  -> std::string
{
  auto path = std::filesystem::path(filename);
  auto stem = path.stem().string();
  auto ext = path.extension().string();
  auto parent = path.parent_path();
  return (parent / (stem + suffix + ext)).string();
}

constexpr std::uint64_t kIbinCacheMetaMagic   = 0x564543464c574341ULL;
constexpr std::uint32_t kIbinCacheMetaVersion = 1;
constexpr std::uint64_t kFnv1a64Offset        = 14695981039346656037ULL;
constexpr std::uint64_t kFnv1a64Prime         = 1099511628211ULL;

struct ibin_cache_meta {
  std::uint64_t magic          = kIbinCacheMetaMagic;
  std::uint32_t version        = kIbinCacheMetaVersion;
  std::uint32_t data_type_size = 0;
  std::int64_t rows            = 0;
  std::int64_t cols            = 0;
  std::uint64_t payload_checksum = 0;
};

inline auto ibin_cache_meta_path(const std::string& filename) -> std::string
{
  return filename + ".meta";
}

inline auto fnv1a64_update(std::uint64_t checksum, const void* data, std::size_t bytes)
  -> std::uint64_t
{
  auto* ptr = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    checksum ^= static_cast<std::uint64_t>(ptr[i]);
    checksum *= kFnv1a64Prime;
  }
  return checksum;
}

inline void write_ibin_cache_meta(const std::string& filename, const ibin_cache_meta& meta)
{
  std::ofstream file(ibin_cache_meta_path(filename), std::ios::binary | std::ios::trunc);
  if (!file) {
    throw std::runtime_error("Cannot create ibin cache meta file: " + ibin_cache_meta_path(filename));
  }
  file.write(reinterpret_cast<const char*>(&meta.magic), sizeof(meta.magic));
  file.write(reinterpret_cast<const char*>(&meta.version), sizeof(meta.version));
  file.write(reinterpret_cast<const char*>(&meta.data_type_size), sizeof(meta.data_type_size));
  file.write(reinterpret_cast<const char*>(&meta.rows), sizeof(meta.rows));
  file.write(reinterpret_cast<const char*>(&meta.cols), sizeof(meta.cols));
  file.write(reinterpret_cast<const char*>(&meta.payload_checksum), sizeof(meta.payload_checksum));
  if (!file) {
    throw std::runtime_error("Cannot write ibin cache meta file: " + ibin_cache_meta_path(filename));
  }
}

inline auto read_ibin_cache_meta(const std::string& filename) -> std::optional<ibin_cache_meta>
{
  auto meta_path = ibin_cache_meta_path(filename);
  if (!std::filesystem::exists(meta_path)) { return std::nullopt; }

  std::ifstream file(meta_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open ibin cache meta file: " + meta_path);
  }

  ibin_cache_meta meta;
  file.read(reinterpret_cast<char*>(&meta.magic), sizeof(meta.magic));
  file.read(reinterpret_cast<char*>(&meta.version), sizeof(meta.version));
  file.read(reinterpret_cast<char*>(&meta.data_type_size), sizeof(meta.data_type_size));
  file.read(reinterpret_cast<char*>(&meta.rows), sizeof(meta.rows));
  file.read(reinterpret_cast<char*>(&meta.cols), sizeof(meta.cols));
  file.read(reinterpret_cast<char*>(&meta.payload_checksum), sizeof(meta.payload_checksum));
  if (!file) {
    throw std::runtime_error("Cannot read ibin cache meta file: " + meta_path);
  }
  return meta;
}

inline void validate_ibin_cache_meta(const std::string& filename,
                                     std::optional<std::uint32_t> expected_type_size = std::nullopt)
{
  auto cache_key = filename + "#" +
                   (expected_type_size.has_value() ? std::to_string(*expected_type_size) : std::string("*"));
  struct validated_entry {
    std::filesystem::file_time_type mtime;
    std::uintmax_t file_size;
  };
  thread_local std::unordered_map<std::string, validated_entry> validated;
  if (auto it = validated.find(cache_key); it != validated.end()) {
    std::error_code ec;
    auto mtime = std::filesystem::last_write_time(filename, ec);
    if (!ec) {
      auto fsize = std::filesystem::file_size(filename, ec);
      if (!ec && it->second.mtime == mtime && it->second.file_size == fsize) {
        return;
      }
    }
    validated.erase(it);
  }

  auto meta_opt = read_ibin_cache_meta(filename);
  if (!meta_opt.has_value()) {
    throw std::runtime_error("Missing ibin cache meta file for: " + filename);
  }

  auto meta = *meta_opt;
  if (meta.magic != kIbinCacheMetaMagic || meta.version != kIbinCacheMetaVersion) {
    throw std::runtime_error("Invalid ibin cache meta header for: " + filename);
  }
  if (expected_type_size.has_value() && meta.data_type_size != *expected_type_size) {
    throw std::runtime_error("ibin cache meta data type size mismatch for: " + filename);
  }

  std::ifstream file(filename, std::ios::binary);
  if (!file) { throw std::runtime_error("Cannot open file: " + filename); }

  std::int64_t rows = 0;
  std::int64_t cols = 0;
  file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  if (!file) { throw std::runtime_error("Cannot read ibin header from: " + filename); }
  if (rows != meta.rows || cols != meta.cols) {
    throw std::runtime_error("ibin cache meta shape mismatch for: " + filename);
  }

  std::array<char, 1 << 20> buffer{};
  auto checksum = kFnv1a64Offset;
  while (file) {
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    auto bytes = static_cast<std::size_t>(file.gcount());
    if (bytes == 0) { break; }
    checksum = fnv1a64_update(checksum, buffer.data(), bytes);
  }
  if (file.bad()) {
    throw std::runtime_error("Cannot read ibin payload from: " + filename);
  }
  if (checksum != meta.payload_checksum) {
    throw std::runtime_error("ibin cache meta checksum mismatch for: " + filename);
  }

  {
    std::error_code ec;
    auto mtime = std::filesystem::last_write_time(filename, ec);
    auto fsize = ec ? std::uintmax_t{0} : std::filesystem::file_size(filename, ec);
    if (!ec) {
      validated[cache_key] = validated_entry{mtime, fsize};
    }
  }
}

inline auto read_ibin_shape(const std::string& filename) -> std::pair<int64_t, int64_t>
{
  std::ifstream file(filename, std::ios::binary);
  if (!file) { throw std::runtime_error("Cannot open file: " + filename); }

  int64_t rows = 0;
  int64_t cols = 0;
  file.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
  file.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));
  if (!file) { throw std::runtime_error("Cannot read ibin header from: " + filename); }
  return {rows, cols};
}

inline void save_matrix_to_ibin(const std::string& filename,
                                raft::host_matrix_view<uint32_t, int64_t> matrix) {
  
  int64_t rows = matrix.extent(0);
  int64_t cols = matrix.extent(1);
  std::ofstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot create file: " + filename);

  file.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));
  file.write(reinterpret_cast<const char*>(matrix.data_handle()), rows * cols * sizeof(uint32_t));
  if (!file) { throw std::runtime_error("Cannot write file: " + filename); }
  file.close();
  if (!file) { throw std::runtime_error("Cannot finalize file: " + filename); }

  auto payload_checksum =
    fnv1a64_update(kFnv1a64Offset,
                   matrix.data_handle(),
                   static_cast<std::size_t>(rows * cols) * sizeof(uint32_t));
  write_ibin_cache_meta(filename,
                        ibin_cache_meta{kIbinCacheMetaMagic,
                                        kIbinCacheMetaVersion,
                                        static_cast<std::uint32_t>(sizeof(uint32_t)),
                                        rows,
                                        cols,
                                        payload_checksum});
  std::cout << "Saving graph to " << filename << std::endl;
}

inline void load_matrix_from_ibin(const std::string& filename,
                                  raft::host_matrix_view<uint32_t, int64_t> matrix) {
  
  std::ifstream file(filename, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot open file: " + filename);

  int64_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(int64_t));
  file.read(reinterpret_cast<char*>(&cols), sizeof(int64_t));

  if (rows != matrix.extent(0) || cols != matrix.extent(1))
    throw std::runtime_error("File dimensions do not match pre-allocated graph dimensions");

  file.read(reinterpret_cast<char*>(matrix.data_handle()), rows * cols * sizeof(uint32_t));
  file.close();
  std::cout << "Loading graph from " << filename << std::endl;
}

inline void save_vector_to_ibin(const std::string& filename, const std::vector<uint32_t>& values)
{
  std::ofstream file(filename, std::ios::binary);
  if (!file) { throw std::runtime_error("Cannot create file: " + filename); }

  auto rows = static_cast<int64_t>(values.size());
  auto cols = int64_t{1};
  file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  file.write(reinterpret_cast<const char*>(values.data()), rows * sizeof(uint32_t));
  if (!file) { throw std::runtime_error("Cannot write vector to file: " + filename); }
}

inline auto load_vector_from_ibin(const std::string& filename) -> std::vector<uint32_t>
{
  auto [rows, cols] = read_ibin_shape(filename);
  if (cols != 1) {
    throw std::runtime_error("Vector cache file does not have one column: " + filename);
  }

  std::ifstream file(filename, std::ios::binary);
  if (!file) { throw std::runtime_error("Cannot open file: " + filename); }
  file.seekg(sizeof(int64_t) * 2, std::ios::beg);

  std::vector<uint32_t> values(static_cast<std::size_t>(rows));
  file.read(reinterpret_cast<char*>(values.data()), rows * sizeof(uint32_t));
  if (!file) { throw std::runtime_error("Cannot read vector cache file: " + filename); }
  return values;
}

template<typename T>
struct QueryInfo {
  raft::device_vector<uint32_t, int64_t> cagra_query_map;
  raft::device_matrix<T, int64_t> cagra_queries;
  raft::device_vector<uint32_t, int64_t> cagra_query_labels;
  raft::device_vector<uint32_t, int64_t> bfs_query_map;
  raft::device_matrix<T, int64_t> bfs_queries;
  raft::device_vector<uint32_t, int64_t> bfs_query_labels;
};

template <typename T>
struct query_classification_scratch {
  explicit query_classification_scratch(cudaStream_t stream)
    : temp_cagra_map(0, stream),
      temp_bfs_map(0, stream),
      temp_cagra_labels(0, stream),
      temp_bfs_labels(0, stream),
      counters(2, stream)
  {
  }

  void ensure_capacity(std::size_t requested_queries, cudaStream_t stream)
  {
    if (capacity_queries >= requested_queries) { return; }
    temp_cagra_map.resize(requested_queries, stream);
    temp_bfs_map.resize(requested_queries, stream);
    temp_cagra_labels.resize(requested_queries, stream);
    temp_bfs_labels.resize(requested_queries, stream);
    capacity_queries = requested_queries;
  }

  void reset(cudaStream_t stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(counters.data(), 0, counters.size() * sizeof(int), stream));
  }

  rmm::device_uvector<uint32_t> temp_cagra_map;
  rmm::device_uvector<uint32_t> temp_bfs_map;
  rmm::device_uvector<uint32_t> temp_cagra_labels;
  rmm::device_uvector<uint32_t> temp_bfs_labels;
  rmm::device_uvector<int> counters;
  std::size_t capacity_queries = 0;
};

static __global__ void classify_queries_kernel(uint32_t* query_labels,
                                               uint32_t* cat_freq,
                                               uint32_t* temp_cagra_map,
                                               uint32_t* temp_bfs_map,
                                               uint32_t* temp_cagra_labels,
                                               uint32_t* temp_bfs_labels,
                                               int n_queries,
                                               int n_labels,
                                               int specificity_threshold,
                                               int* cagra_count,
                                               int* bfs_count) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_queries) return;

  uint32_t label = query_labels[tid];
  if (label >= static_cast<uint32_t>(n_labels)) return;
  uint32_t freq = cat_freq[label];
  bool is_cagra = freq > specificity_threshold;
  
  int pos;
  if (is_cagra) {
    pos = atomicAdd(cagra_count, 1);
    temp_cagra_map[pos] = tid;
    temp_cagra_labels[pos] = label;
  } else {
    pos = atomicAdd(bfs_count, 1);
    temp_bfs_map[pos] = tid;
    temp_bfs_labels[pos] = label;
  }
}

template <typename T>
__global__ void gather_queries_by_map_kernel(const T* queries,
                                             const uint32_t* query_map,
                                             T* gathered_queries,
                                             int64_t n_queries,
                                             int64_t dim)
{
  auto tid   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = n_queries * dim;
  if (tid >= total) { return; }

  auto row = tid / dim;
  auto col = tid % dim;
  gathered_queries[tid] = queries[static_cast<int64_t>(query_map[row]) * dim + col];
}

template<typename T>
inline auto classify_queries(raft::resources const& res,
                             raft::device_matrix_view<const T, int64_t> queries,
                             raft::device_vector_view<uint32_t, int64_t> query_labels,
                             raft::device_vector_view<uint32_t, int64_t> cat_freq,
                             int specificity_threshold,
                             query_classification_scratch<T>* scratch = nullptr) -> QueryInfo<T> {
  
  int n_queries = queries.extent(0);
  int dim = queries.extent(1);

  if (n_queries == 0) {
    return QueryInfo<T>{
      raft::make_device_vector<uint32_t, int64_t>(res, 0),
      raft::make_device_matrix<T, int64_t>(res, 0, dim),
      raft::make_device_vector<uint32_t, int64_t>(res, 0),
      raft::make_device_vector<uint32_t, int64_t>(res, 0),
      raft::make_device_matrix<T, int64_t>(res, 0, dim),
      raft::make_device_vector<uint32_t, int64_t>(res, 0)};
  }

  auto stream = raft::resource::get_cuda_stream(res);

  std::unique_ptr<query_classification_scratch<T>> owned_scratch;
  if (scratch == nullptr) {
    owned_scratch = std::make_unique<query_classification_scratch<T>>(stream);
    scratch = owned_scratch.get();
  }
  scratch->ensure_capacity(static_cast<std::size_t>(n_queries), stream);
  scratch->reset(stream);
  
  // Launch kernel
  int block_size = 256;
  int grid_size = (n_queries + block_size - 1) / block_size;
  
  classify_queries_kernel<<<grid_size, block_size, 0, stream>>>(
    query_labels.data_handle(),
    cat_freq.data_handle(),
    scratch->temp_cagra_map.data(),
    scratch->temp_bfs_map.data(),
    scratch->temp_cagra_labels.data(),
    scratch->temp_bfs_labels.data(),
    n_queries,
    static_cast<int>(cat_freq.extent(0)),
    specificity_threshold,
    scratch->counters.data(),
    scratch->counters.data() + 1
  );
  
  // Get final counts
  int host_counts[2] = {0, 0};
  raft::copy(host_counts, scratch->counters.data(), 2, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  int h_cagra_count = host_counts[0];
  int h_bfs_count = host_counts[1];

  // Initialize raft structures with correct sizes
  auto cagra_query_map = raft::make_device_vector<uint32_t, int64_t>(res, h_cagra_count);
  auto cagra_queries = raft::make_device_matrix<T, int64_t>(res, h_cagra_count, dim);
  auto cagra_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, h_cagra_count);
  auto bfs_query_map = raft::make_device_vector<uint32_t, int64_t>(res, h_bfs_count);
  auto bfs_queries = raft::make_device_matrix<T, int64_t>(res, h_bfs_count, dim);
  auto bfs_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, h_bfs_count);
  
  // Copy from temporary buffers to final raft structures
  raft::copy(cagra_query_map.data_handle(),
             scratch->temp_cagra_map.data(),
             h_cagra_count,
             stream);
  raft::copy(cagra_query_labels.data_handle(),
             scratch->temp_cagra_labels.data(),
             h_cagra_count,
             stream);
  
  raft::copy(bfs_query_map.data_handle(),
             scratch->temp_bfs_map.data(),
             h_bfs_count,
             stream);
  raft::copy(bfs_query_labels.data_handle(),
             scratch->temp_bfs_labels.data(),
             h_bfs_count,
             stream);

  if (h_cagra_count > 0) {
    auto cagra_total_values = static_cast<int64_t>(h_cagra_count) * dim;
    auto cagra_grid_size =
      static_cast<int>((cagra_total_values + block_size - 1) / block_size);
    gather_queries_by_map_kernel<<<cagra_grid_size, block_size, 0, stream>>>(
      queries.data_handle(),
      cagra_query_map.data_handle(),
      cagra_queries.data_handle(),
      h_cagra_count,
      dim);
  }

  if (h_bfs_count > 0) {
    auto bfs_total_values = static_cast<int64_t>(h_bfs_count) * dim;
    auto bfs_grid_size =
      static_cast<int>((bfs_total_values + block_size - 1) / block_size);
    gather_queries_by_map_kernel<<<bfs_grid_size, block_size, 0, stream>>>(
      queries.data_handle(),
      bfs_query_map.data_handle(),
      bfs_queries.data_handle(),
      h_bfs_count,
      dim);
  }
  
  return QueryInfo<T> {
    std::move(cagra_query_map),
    std::move(cagra_queries),
    std::move(cagra_query_labels),
    std::move(bfs_query_map),
    std::move(bfs_queries),
    std::move(bfs_query_labels)
  };
}

template <typename T>
__global__ void fill_values_kernel(T* output, int64_t size, T value)
{
  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= size) { return; }
  output[tid] = value;
}

template <typename T>
inline void fill_device_values(raft::resources const& res, T* output, int64_t size, T value)
{
  if (size <= 0) { return; }
  auto stream = raft::resource::get_cuda_stream(res);
  auto block_size = 256;
  auto grid_size = static_cast<int>((size + block_size - 1) / block_size);
  fill_values_kernel<<<grid_size, block_size, 0, stream>>>(output, size, value);
}

inline void initialize_search_results(raft::resources const& res,
                                      raft::device_matrix_view<uint32_t, int64_t> neighbors,
                                      raft::device_matrix_view<float, int64_t> distances)
{
  auto stream = raft::resource::get_cuda_stream(res);
  RAFT_CUDA_TRY(cudaMemsetAsync(
    neighbors.data_handle(), 0xFF, neighbors.size() * sizeof(uint32_t), stream));
  fill_device_values<float>(
    res, distances.data_handle(), distances.size(), std::numeric_limits<float>::infinity());
}

template<typename T, typename IdxT>
__global__ void merge_neighbors_kernel(uint32_t* neighbors,
                                       float* distances,
                                       const IdxT* neighbor_src,
                                       const float* distance_src,
                                       const uint32_t* indices,
                                       int n_queries,
                                       int topk) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n_elements = n_queries * topk;
  
  if (tid >= n_elements) return;
  
  int query_idx = tid / topk;
  int k_idx = tid % topk;
  
  int src_offset = query_idx * topk + k_idx;
  int dst_offset = indices[query_idx] * topk + k_idx;
  
  neighbors[dst_offset] = static_cast<uint32_t>(neighbor_src[src_offset]);
  distances[dst_offset] = distance_src[src_offset];
}

template<typename T>
inline void merge_search_results(raft::resources const& res,
                                 raft::device_matrix_view<uint32_t, int64_t> neighbors,
                                 raft::device_matrix_view<float, int64_t> distances,
                                 QueryInfo<T>& query_info,
                                 raft::device_matrix_view<int64_t, int64_t> bfs_neighbors,
                                 raft::device_matrix_view<float, int64_t> bfs_distances,
                                 raft::device_matrix_view<uint32_t, int64_t> cagra_neighbors,
                                 raft::device_matrix_view<float, int64_t> cagra_distances,
                                 int topk) {
  
  auto stream = raft::resource::get_cuda_stream(res);
  
  // Launch kernels for both BFS and CAGRA results
  int block_size = 256;
  
  if (query_info.bfs_query_map.size() > 0) {
    // BFS kernel - handles conversion and merging in one step
    int n_bfs_elements = query_info.bfs_query_map.size() * topk;
    int grid_size_bfs = (n_bfs_elements + block_size - 1) / block_size;
    
    merge_neighbors_kernel<T, int64_t><<<grid_size_bfs, block_size, 0, stream>>>(
      neighbors.data_handle(),
      distances.data_handle(),
      bfs_neighbors.data_handle(),  // Direct use of int64_t input
      bfs_distances.data_handle(),
      query_info.bfs_query_map.data_handle(),
      query_info.bfs_query_map.size(),
      topk);
  }
  
  if (query_info.cagra_query_map.size() > 0) {
    // CAGRA kernel
    int n_cagra_elements = query_info.cagra_query_map.size() * topk;
    int grid_size_cagra = (n_cagra_elements + block_size - 1) / block_size;
    
    merge_neighbors_kernel<T, uint32_t><<<grid_size_cagra, block_size, 0, stream>>>(
      neighbors.data_handle(),
      distances.data_handle(),
      cagra_neighbors.data_handle(),
      cagra_distances.data_handle(),
      query_info.cagra_query_map.data_handle(),
      query_info.cagra_query_map.size(),
      topk);
  }
}

} // namespace cuvs::neighbors::vecflow
