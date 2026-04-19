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

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/filtered_bfs.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <memory>
#include <atomic>
#include <list>
#include <mutex>
#include <optional>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuvs::neighbors::vecflow {

template <typename data_t>
struct query_classification_scratch;

enum class graph_builder_type : uint8_t {
  CAGRA = 0,
  TAGORE_CAGRA_COMPAT = 1,
  TAGORE_CAGRA = TAGORE_CAGRA_COMPAT
};

struct phoenix_label_cache_state {
  struct cached_graph_entry {
    std::shared_ptr<uint32_t> storage;
    std::size_t bytes = 0;
    std::list<uint32_t>::iterator lru_it;
  };

  std::mutex mutex;
  std::unordered_map<uint32_t, cached_graph_entry> graphs;
  std::list<uint32_t> lru_labels;
  std::size_t cached_bytes = 0;
  std::unordered_map<uint32_t, cached_graph_entry> host_graphs;
  std::list<uint32_t> host_lru_labels;
  std::size_t host_cached_bytes = 0;
  std::vector<std::uint64_t> access_counts;
  std::uint64_t access_events = 0;
  std::uint64_t hbm_hits       = 0;
  std::uint64_t dram_hits      = 0;
  std::uint64_t ssd_loads      = 0;
  std::uint64_t hbm_evictions  = 0;
  std::uint64_t dram_evictions = 0;
};

template <typename data_t>
struct phoenix_label_dataset_cache_state {
  struct cached_dataset_entry {
    std::shared_ptr<data_t> storage;
    std::size_t bytes = 0;
    std::list<uint32_t>::iterator lru_it;
  };

  std::mutex mutex;
  std::unordered_map<uint32_t, cached_dataset_entry> datasets;
  std::list<uint32_t> lru_labels;
  std::size_t cached_bytes = 0;
  std::unordered_map<uint32_t, cached_dataset_entry> host_datasets;
  std::list<uint32_t> host_lru_labels;
  std::size_t host_cached_bytes = 0;
  std::uint64_t hbm_hits        = 0;
  std::uint64_t dram_hits       = 0;
  std::uint64_t ssd_loads       = 0;
  std::uint64_t hbm_evictions   = 0;
  std::uint64_t dram_evictions  = 0;
};

template <typename data_t>
struct bfs_label_cache_state {
  struct cached_bfs_entry {
    std::shared_ptr<cuvs::neighbors::ivf_flat::index<data_t, int64_t>> index;
    std::shared_ptr<data_t> storage;
    std::size_t bytes = 0;
    int64_t label_size = 0;
    std::list<uint32_t>::iterator lru_it;
  };

  std::mutex hbm_mutex;
  std::mutex dram_mutex;
  std::mutex access_mutex;
  std::unordered_map<uint32_t, cached_bfs_entry> hbm_entries;
  std::list<uint32_t> hbm_lru;
  std::size_t hbm_cached_bytes = 0;
  std::unordered_map<uint32_t, cached_bfs_entry> dram_entries;
  std::list<uint32_t> dram_lru;
  std::size_t dram_cached_bytes = 0;
  std::vector<std::uint64_t> access_counts;
  std::uint64_t access_events = 0;
  std::uint64_t hbm_hits       = 0;
  std::uint64_t dram_hits      = 0;
  std::uint64_t ssd_loads      = 0;
  std::uint64_t hbm_evictions  = 0;
  std::uint64_t dram_evictions = 0;
};

struct multi_label_query_desc {
  enum class combine_mode : uint8_t { OR = 0, AND = 1 };
  enum class and_strategy : uint8_t { GREEDY = 0, PARALLEL = 1 };

  raft::device_vector<int64_t, int64_t> label_offsets;
  raft::device_vector<uint32_t, int64_t> label_indices;
  std::vector<int64_t> host_label_offsets;
  std::vector<uint32_t> host_label_indices;
  combine_mode mode     = combine_mode::OR;
  and_strategy and_mode = and_strategy::GREEDY;
};

/**
 * @brief The vecflow index holds all internal information required for search.
 *
 * This includes the IVF-graph index, the IVF-BFS index, and metadata such as label sizes,
 * offsets, and the mapping of data points for each label.
 *
 */
template <typename data_t>
struct index {
  cuvs::neighbors::cagra::index<data_t, uint32_t> ivf_graph_index;
  cuvs::neighbors::ivf_flat::index<data_t, int64_t> ivf_bfs_index;

  int specificity_threshold = 2000;
  
  raft::device_vector<uint32_t, int64_t> cagra_index_map;
  raft::device_vector<uint32_t, int64_t> cagra_label_size;
  raft::device_vector<uint32_t, int64_t> cagra_label_offset;
  raft::device_vector<uint32_t, int64_t> bfs_label_size;
  raft::device_vector<uint32_t, int64_t> bfs_label_offset;
  raft::device_vector<uint32_t, int64_t> bfs_index_map;
  std::shared_ptr<bfs_label_cache_state<data_t>> bfs_cache =
    std::make_shared<bfs_label_cache_state<data_t>>();
  bool bfs_tiered_cache_enabled = false;
  std::size_t bfs_hbm_capacity_bytes = 0;
  std::size_t bfs_dram_capacity_bytes = 0;
  std::size_t bfs_prefetch_max_bytes = 0;
  std::string bfs_dataset_cache_fname;
  std::vector<uint32_t> host_bfs_index_map;
  std::vector<uint32_t> host_bfs_label_offset;
  std::vector<uint32_t> host_bfs_label_size;
  int64_t bfs_total_rows = 0;
  std::size_t bfs_rebalance_interval_queries = 0;
  raft::device_vector<uint32_t, int64_t> cat_freq;
  std::vector<uint32_t> host_cat_freq;
  std::shared_ptr<uint32_t> cagra_graph_storage;
  std::vector<uint32_t> host_cagra_label_size;
  std::vector<uint32_t> host_cagra_label_offset;
  std::string cagra_graph_cache_fname;
  int cagra_graph_degree = 0;
  int cagra_graph_storage_width = 0;
  std::string cagra_dataset_cache_fname;
  int64_t dataset_rows = 0;
  int dataset_dim = 0;
  graph_builder_type graph_builder = graph_builder_type::CAGRA;
  std::shared_ptr<phoenix_label_cache_state> phoenix_label_cache =
    std::make_shared<phoenix_label_cache_state>();
  std::shared_ptr<phoenix_label_dataset_cache_state<data_t>> phoenix_label_dataset_cache =
    std::make_shared<phoenix_label_dataset_cache_state<data_t>>();
  std::size_t phoenix_label_cache_capacity_bytes = 0;
  std::size_t phoenix_label_dram_cache_capacity_bytes = 0;
  std::size_t phoenix_label_dataset_cache_capacity_bytes = 0;
  std::size_t phoenix_label_dataset_dram_cache_capacity_bytes = 0;
  std::size_t phoenix_rebalance_interval_queries = 0;
  std::shared_ptr<query_classification_scratch<data_t>> query_classification_scratch;
};

struct runtime_config {
  std::optional<bool> use_phoenix_graph_load;
  std::optional<bool> use_phoenix_label_load;
  std::optional<bool> enable_bfs_tiered_cache;
  std::optional<std::size_t> phoenix_label_cache_bytes;
  std::optional<std::size_t> phoenix_label_dram_cache_bytes;
  std::optional<std::size_t> phoenix_label_prefetch_max_bytes;
  std::optional<std::size_t> phoenix_label_rebalance_interval_queries;
  std::optional<std::size_t> phoenix_label_dataset_cache_bytes;
  std::optional<std::size_t> phoenix_label_dataset_dram_cache_bytes;
  std::optional<std::size_t> phoenix_label_dataset_prefetch_max_bytes;
  std::optional<std::size_t> bfs_hbm_cache_bytes;
  std::optional<std::size_t> bfs_dram_cache_bytes;
  std::optional<std::size_t> bfs_prefetch_max_bytes;
  std::optional<std::size_t> bfs_rebalance_interval_queries;
  std::optional<bool> cascade_eviction;
};

namespace detail {

inline auto runtime_config_mutex() -> std::mutex&
{
  static std::mutex mutex;
  return mutex;
}

inline auto mutable_runtime_config_override() -> std::optional<runtime_config>&
{
  static std::optional<runtime_config> config;
  return config;
}

}  // namespace detail

inline void set_runtime_config(runtime_config config)
{
  std::lock_guard<std::mutex> lock(detail::runtime_config_mutex());
  detail::mutable_runtime_config_override() = std::move(config);
}

inline void clear_runtime_config()
{
  std::lock_guard<std::mutex> lock(detail::runtime_config_mutex());
  detail::mutable_runtime_config_override().reset();
}

inline auto get_runtime_config() -> std::optional<runtime_config>
{
  std::lock_guard<std::mutex> lock(detail::runtime_config_mutex());
  return detail::mutable_runtime_config_override();
}

class scoped_runtime_config {
 public:
  explicit scoped_runtime_config(runtime_config config)
    : previous_(get_runtime_config())
  {
    set_runtime_config(std::move(config));
  }

  ~scoped_runtime_config()
  {
    if (previous_.has_value()) {
      set_runtime_config(*previous_);
    } else {
      clear_runtime_config();
    }
  }

  scoped_runtime_config(scoped_runtime_config const&) = delete;
  auto operator=(scoped_runtime_config const&) -> scoped_runtime_config& = delete;

 private:
  std::optional<runtime_config> previous_;
};

struct multi_gpu_params {
  std::vector<int> device_ids;
  double label_routing_query_weight = 1.0;
  double label_routing_data_weight  = 1.0;
  std::vector<double> label_query_weights;
};

template <typename data_t>
struct multi_gpu_index {
  struct worker_context {
    int device_id = 0;
    std::vector<uint32_t> labels;
    std::shared_ptr<shared_resources::configured_raft_resources> resources;
    std::optional<raft::device_matrix<data_t, int64_t>> dataset;
    std::optional<cuvs::neighbors::vecflow::index<data_t>> index;
  };

  std::vector<worker_context> workers;
  std::unordered_map<uint32_t, int> label_to_worker;
  std::vector<double> worker_loads;
  int graph_degree = 0;
  int specificity_threshold = 0;
  graph_builder_type graph_builder = graph_builder_type::CAGRA;
};

struct storage_tier_stats {
  std::size_t labels = 0;
  std::size_t bytes  = 0;
};

struct storage_component_stats {
  storage_tier_stats hbm;
  storage_tier_stats dram;
  storage_tier_stats ssd;
  storage_tier_stats resident_hbm;
  storage_tier_stats resident_dram;
  std::size_t total_labels = 0;
  std::size_t total_bytes  = 0;
};

struct storage_stats_info {
  storage_component_stats graph;
  storage_component_stats dataset;
  storage_component_stats bfs;
  std::size_t access_events = 0;
};

struct multi_gpu_storage_stats_info {
  std::vector<int> worker_device_ids;
  std::vector<std::size_t> worker_owned_labels;
  std::vector<double> worker_loads;
  std::vector<storage_stats_info> worker_storage;
  std::size_t mapped_labels = 0;
};

/**
 * @brief Builds (or loads) the vecflow index.
 *
 * This function builds (or loads from file) both the IVF-graph index and the IVF-BFS index.
 *
 * @param res                   RAFT shared resources.
 * @param d_dataset             Device matrix view of the dataset.
 * @param data_label_vecs       Vector of vectors of data labels.
 * @param graph_degree          Desired graph degree.
 * @param specificity_threshold Threshold to decide which labels go to CAGRA vs. BFS.
 * @param graph_fname           (Optional) File name to load/save the IVF-graph index.
 * @param bfs_fname             (Optional) File name to load/save the BFS index.
 * @param force_rebuild         (Optional) Whether to force rebuild the index.
 */
auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const float, int64_t> d_dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname = "",
           const std::string& bfs_fname = "",
           bool force_rebuild = false,
           graph_builder_type graph_builder = graph_builder_type::CAGRA,
           int tagore_iterations = 10) -> cuvs::neighbors::vecflow::index<float>;

auto build(shared_resources::configured_raft_resources& res,
           raft::device_matrix_view<const int8_t, int64_t> d_dataset,
           const std::vector<std::vector<int>>& data_label_vecs,
           int graph_degree,
           int specificity_threshold,
           const std::string& graph_fname = "",
           const std::string& bfs_fname = "",
           bool force_rebuild = false,
           graph_builder_type graph_builder = graph_builder_type::CAGRA,
           int tagore_iterations = 10) -> cuvs::neighbors::vecflow::index<int8_t>;

auto build_multi_gpu(shared_resources::configured_raft_resources& res,
                     raft::device_matrix_view<const float, int64_t> d_dataset,
                     const std::vector<std::vector<int>>& data_label_vecs,
                     int graph_degree,
                     int specificity_threshold,
                     const multi_gpu_params& mg_params,
                     const std::string& graph_fname = "",
                     const std::string& bfs_fname = "",
                     bool force_rebuild = false,
                     graph_builder_type graph_builder = graph_builder_type::CAGRA,
                     int tagore_iterations = 10) -> cuvs::neighbors::vecflow::multi_gpu_index<float>;
/**
 * @brief Performs a vecflow search.
 *
 * Given a set of queries and query labels, this function searches for the top-k nearest
 * neighbors using the vecflow index.
 *
 * @param res           RAFT shared resources.
 * @param index         The vecflow index to use.
 * @param queries       Device matrix view of query vectors.
 * @param query_labels  Device vector view of query labels.
 * @param itopk_size    Number of top results to return.
 * @param neighbors     [out] Device matrix view to hold neighbor indices.
 * @param distances     [out] Device matrix view to hold distances.
 */
void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<float>& index,
            raft::device_matrix_view<const float, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances);

void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<float>& index,
            raft::device_matrix_view<const float, int64_t> queries,
            const multi_label_query_desc& query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances);

void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<int8_t>& index,
            raft::device_matrix_view<const int8_t, int64_t> queries,
            raft::device_vector_view<uint32_t, int64_t> query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances);

void search(shared_resources::configured_raft_resources& res,
            cuvs::neighbors::vecflow::index<int8_t>& index,
            raft::device_matrix_view<const int8_t, int64_t> queries,
            const multi_label_query_desc& query_labels,
            int itopk_size,
            raft::device_matrix_view<uint32_t, int64_t> neighbors,
            raft::device_matrix_view<float, int64_t> distances);

void search_multi_gpu(shared_resources::configured_raft_resources& res,
                      cuvs::neighbors::vecflow::multi_gpu_index<float>& index,
                      raft::device_matrix_view<const float, int64_t> queries,
                      raft::device_vector_view<uint32_t, int64_t> query_labels,
                      int itopk_size,
                      raft::device_matrix_view<uint32_t, int64_t> neighbors,
                      raft::device_matrix_view<float, int64_t> distances);

namespace detail {

template <typename data_t>
inline auto graph_label_bytes(const cuvs::neighbors::vecflow::index<data_t>& index,
                              uint32_t label_size) -> std::size_t
{
  auto graph_width =
    index.cagra_graph_storage_width > 0 ? index.cagra_graph_storage_width : index.cagra_graph_degree;
  if (graph_width <= 0 || label_size == 0) { return 0; }
  return static_cast<std::size_t>(label_size) * static_cast<std::size_t>(graph_width) *
         sizeof(uint32_t);
}

template <typename data_t>
inline auto dataset_label_bytes(const cuvs::neighbors::vecflow::index<data_t>& index,
                                uint32_t label_size) -> std::size_t
{
  auto dim = index.dataset_dim > 0
               ? index.dataset_dim
               : static_cast<int>(index.ivf_graph_index.dataset().extent(1));
  if (dim <= 0 || label_size == 0) { return 0; }
  return static_cast<std::size_t>(label_size) * static_cast<std::size_t>(dim) * sizeof(data_t);
}

template <typename data_t>
inline auto bfs_hbm_label_bytes(const cuvs::neighbors::vecflow::index<data_t>& index,
                                uint32_t label_size) -> std::size_t
{
  if (index.dataset_dim <= 0 || label_size == 0) { return 0; }
  auto group_size = static_cast<std::size_t>(cuvs::neighbors::ivf_flat::kIndexGroupSize);
  auto label_size_us = static_cast<std::size_t>(label_size);
  auto padded_size = ((label_size_us + group_size - 1) / group_size) * group_size;
  return padded_size * static_cast<std::size_t>(index.dataset_dim) * sizeof(data_t) +
         padded_size * sizeof(int64_t);
}

template <typename data_t>
inline auto bfs_dram_label_bytes(const cuvs::neighbors::vecflow::index<data_t>& index,
                                 uint32_t label_size) -> std::size_t
{
  return dataset_label_bytes(index, label_size);
}

template <typename data_t>
inline auto bfs_ssd_label_bytes(const cuvs::neighbors::vecflow::index<data_t>& index,
                                uint32_t label_size) -> std::size_t
{
  return bfs_dram_label_bytes(index, label_size);
}

template <typename entry_t>
inline void fill_resident_cache_stats(const std::unordered_map<uint32_t, entry_t>& cache,
                                      std::size_t cached_bytes,
                                      storage_tier_stats* stats,
                                      std::unordered_set<uint32_t>* labels)
{
  if (stats == nullptr || labels == nullptr) { return; }
  stats->labels = cache.size();
  stats->bytes  = cached_bytes;
  labels->reserve(cache.size());
  for (auto const& [label, _] : cache) {
    (void)_;
    labels->insert(label);
  }
}

template <typename data_t>
inline void assign_primary_storage_tiers(const cuvs::neighbors::vecflow::index<data_t>& index,
                                         std::size_t total_labels,
                                         std::size_t total_bytes,
                                         bool fully_attached_in_hbm,
                                         const std::unordered_set<uint32_t>& hbm_labels,
                                         const std::unordered_set<uint32_t>& dram_labels,
                                         storage_component_stats* stats,
                                         bool is_dataset)
{
  if (stats == nullptr) { return; }

  stats->total_labels = total_labels;
  stats->total_bytes  = total_bytes;
  if (total_labels == 0) { return; }

  if (fully_attached_in_hbm) {
    stats->hbm.labels  = total_labels;
    stats->hbm.bytes   = total_bytes;
    if (stats->resident_hbm.labels == 0 && stats->resident_hbm.bytes == 0) {
      stats->resident_hbm = stats->hbm;
    }
    return;
  }

  for (uint32_t label = 0; label < index.host_cagra_label_size.size(); ++label) {
    auto label_size = index.host_cagra_label_size[label];
    if (label_size == 0) { continue; }
    auto bytes = is_dataset ? dataset_label_bytes(index, label_size) : graph_label_bytes(index, label_size);
    if (hbm_labels.find(label) != hbm_labels.end()) {
      stats->hbm.labels += 1;
      stats->hbm.bytes += bytes;
    } else if (dram_labels.find(label) != dram_labels.end()) {
      stats->dram.labels += 1;
      stats->dram.bytes += bytes;
    } else {
      stats->ssd.labels += 1;
      stats->ssd.bytes += bytes;
    }
  }
}

}  // namespace detail

template <typename data_t>
inline auto storage_stats(const cuvs::neighbors::vecflow::index<data_t>& index)
  -> storage_stats_info
{
  storage_stats_info stats;

  std::size_t total_graph_labels   = 0;
  std::size_t total_graph_bytes    = 0;
  std::size_t total_dataset_labels = 0;
  std::size_t total_dataset_bytes  = 0;
  std::size_t total_bfs_labels     = 0;
  for (auto label_size : index.host_cagra_label_size) {
    if (label_size == 0) { continue; }
    total_graph_labels += 1;
    total_graph_bytes += detail::graph_label_bytes(index, label_size);
    total_dataset_labels += 1;
    total_dataset_bytes += detail::dataset_label_bytes(index, label_size);
  }
  for (auto label_size : index.host_bfs_label_size) {
    if (label_size == 0) { continue; }
    total_bfs_labels += 1;
  }

  std::unordered_set<uint32_t> graph_hbm_labels;
  std::unordered_set<uint32_t> graph_dram_labels;
  if (index.phoenix_label_cache != nullptr) {
    std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
    detail::fill_resident_cache_stats(index.phoenix_label_cache->graphs,
                                      index.phoenix_label_cache->cached_bytes,
                                      &stats.graph.resident_hbm,
                                      &graph_hbm_labels);
    detail::fill_resident_cache_stats(index.phoenix_label_cache->host_graphs,
                                      index.phoenix_label_cache->host_cached_bytes,
                                      &stats.graph.resident_dram,
                                      &graph_dram_labels);
    stats.access_events += index.phoenix_label_cache->access_events;
  }

  std::unordered_set<uint32_t> dataset_hbm_labels;
  std::unordered_set<uint32_t> dataset_dram_labels;
  if (index.phoenix_label_dataset_cache != nullptr) {
    std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
    detail::fill_resident_cache_stats(index.phoenix_label_dataset_cache->datasets,
                                      index.phoenix_label_dataset_cache->cached_bytes,
                                      &stats.dataset.resident_hbm,
                                      &dataset_hbm_labels);
    detail::fill_resident_cache_stats(index.phoenix_label_dataset_cache->host_datasets,
                                      index.phoenix_label_dataset_cache->host_cached_bytes,
                                      &stats.dataset.resident_dram,
                                      &dataset_dram_labels);
  }

  std::unordered_set<uint32_t> bfs_hbm_labels;
  std::unordered_set<uint32_t> bfs_dram_labels;
  if (index.bfs_cache != nullptr) {
    std::scoped_lock lock(index.bfs_cache->hbm_mutex,
                          index.bfs_cache->dram_mutex,
                          index.bfs_cache->access_mutex);
    detail::fill_resident_cache_stats(index.bfs_cache->hbm_entries,
                                      index.bfs_cache->hbm_cached_bytes,
                                      &stats.bfs.resident_hbm,
                                      &bfs_hbm_labels);
    detail::fill_resident_cache_stats(index.bfs_cache->dram_entries,
                                      index.bfs_cache->dram_cached_bytes,
                                      &stats.bfs.resident_dram,
                                      &bfs_dram_labels);
    stats.access_events += index.bfs_cache->access_events;
  }

  auto graph_fully_attached_in_hbm =
    total_graph_labels > 0 && index.ivf_graph_index.graph().extent(0) > 0 &&
    graph_hbm_labels.empty() && graph_dram_labels.empty();
  auto dataset_fully_attached_in_hbm =
    total_dataset_labels > 0 && index.ivf_graph_index.dataset().extent(0) > 0 &&
    dataset_hbm_labels.empty() && dataset_dram_labels.empty();

  detail::assign_primary_storage_tiers(index,
                                       total_graph_labels,
                                       total_graph_bytes,
                                       graph_fully_attached_in_hbm,
                                       graph_hbm_labels,
                                       graph_dram_labels,
                                       &stats.graph,
                                       false);
  detail::assign_primary_storage_tiers(index,
                                       total_dataset_labels,
                                       total_dataset_bytes,
                                       dataset_fully_attached_in_hbm,
                                       dataset_hbm_labels,
                                       dataset_dram_labels,
                                       &stats.dataset,
                                       true);

  stats.bfs.total_labels = total_bfs_labels;
  if (total_bfs_labels > 0) {
    if (!index.bfs_tiered_cache_enabled) {
      for (auto label_size : index.host_bfs_label_size) {
        if (label_size == 0) { continue; }
        stats.bfs.hbm.labels += 1;
        stats.bfs.hbm.bytes += detail::bfs_hbm_label_bytes(index, label_size);
      }
      stats.bfs.resident_hbm = stats.bfs.hbm;
    } else {
      for (uint32_t label = 0; label < index.host_bfs_label_size.size(); ++label) {
        auto label_size = index.host_bfs_label_size[label];
        if (label_size == 0) { continue; }
        if (bfs_hbm_labels.find(label) != bfs_hbm_labels.end()) {
          stats.bfs.hbm.labels += 1;
          stats.bfs.hbm.bytes += detail::bfs_hbm_label_bytes(index, label_size);
        } else if (bfs_dram_labels.find(label) != bfs_dram_labels.end()) {
          stats.bfs.dram.labels += 1;
          stats.bfs.dram.bytes += detail::bfs_dram_label_bytes(index, label_size);
        } else {
          stats.bfs.ssd.labels += 1;
          stats.bfs.ssd.bytes += detail::bfs_ssd_label_bytes(index, label_size);
        }
      }
    }
  }
  stats.bfs.total_bytes = stats.bfs.hbm.bytes + stats.bfs.dram.bytes + stats.bfs.ssd.bytes;
  return stats;
}

template <typename data_t>
inline auto storage_stats(const cuvs::neighbors::vecflow::multi_gpu_index<data_t>& index)
  -> multi_gpu_storage_stats_info
{
  multi_gpu_storage_stats_info stats;
  stats.worker_loads = index.worker_loads;
  stats.worker_device_ids.reserve(index.workers.size());
  stats.worker_owned_labels.reserve(index.workers.size());
  stats.worker_storage.reserve(index.workers.size());

  for (auto const& worker : index.workers) {
    stats.worker_device_ids.push_back(worker.device_id);
    stats.worker_owned_labels.push_back(worker.labels.size());
    stats.mapped_labels += worker.labels.size();
    if (worker.index.has_value()) {
      stats.worker_storage.push_back(storage_stats(*worker.index));
    } else {
      stats.worker_storage.emplace_back();
    }
  }

  return stats;
}

} // namespace cuvs::neighbors::vecflow
