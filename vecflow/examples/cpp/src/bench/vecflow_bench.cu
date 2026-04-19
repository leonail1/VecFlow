#include <cuvs/neighbors/vecflow.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/shared_resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <chrono>
#include <nlohmann/json.hpp>
#include <atomic>
#include <bitset>
#include <cctype>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <string>

#ifdef VECFLOW_BENCH_PROGRESS_LOG
#include <cstdio>
#include <cinttypes>
#endif

#include "../common.cuh"

using namespace cuvs::neighbors;
using json = nlohmann::json;

struct vecflow_mg_device_context {
  int device_id = 0;
  std::optional<raft::device_matrix<float, int64_t>> dataset;
  std::optional<vecflow::index<float>> index;
};

struct query_chunk {
  int64_t offset = 0;
  int64_t size = 0;
};

std::vector<query_chunk> split_query_chunks(int64_t n_queries, int n_parts)
{
  std::vector<query_chunk> chunks;
  chunks.reserve(n_parts);
  int64_t base = n_queries / n_parts;
  int64_t rem = n_queries % n_parts;
  int64_t offset = 0;
  for (int i = 0; i < n_parts; ++i) {
    int64_t size = base + (i < rem ? 1 : 0);
    chunks.push_back(query_chunk{offset, size});
    offset += size;
  }
  return chunks;
}

std::string append_suffix_to_filename(const std::string& filename, const std::string& suffix)
{
  auto path = std::filesystem::path(filename);
  auto stem = path.stem().string();
  auto ext = path.extension().string();
  auto parent = path.parent_path();
  return (parent / (stem + suffix + ext)).string();
}

std::string sanitize_filename(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return std::isalnum(ch) ? static_cast<char>(ch) : '_';
  });
  return value;
}

std::vector<int64_t> read_query_id_list(const std::string& path)
{
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to open query id list: " + path);
  }

  std::vector<int64_t> ids;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) { continue; }
    ids.push_back(std::stoll(line));
  }
  return ids;
}

std::unordered_set<int> read_allowed_labels_file(const std::string& path)
{
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to open allowed labels file: " + path);
  }

  std::unordered_set<int> labels;
  std::string line;
  while (std::getline(input, line)) {
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
      auto trimmed = token;
      auto first = trimmed.find_first_not_of(" \t\r\n");
      if (first == std::string::npos) { continue; }
      auto last = trimmed.find_last_not_of(" \t\r\n");
      trimmed = trimmed.substr(first, last - first + 1);
      if (trimmed.empty()) { continue; }
      labels.insert(std::stoi(trimmed));
    }
  }
  return labels;
}

void filter_labels_in_place(std::vector<std::vector<int>>& row_labels,
                            const std::unordered_set<int>& allowed_labels)
{
  for (auto& labels : row_labels) {
    labels.erase(std::remove_if(labels.begin(),
                                labels.end(),
                                [&](int label) { return allowed_labels.count(label) == 0; }),
                 labels.end());
  }
}

void rebuild_label_data_mapping(const std::vector<std::vector<int>>& data_label_vecs,
                                std::vector<std::vector<int>>* label_data_vecs)
{
  auto max_label = -1;
  for (auto const& labels : data_label_vecs) {
    for (auto label : labels) {
      if (label > max_label) { max_label = label; }
    }
  }

  label_data_vecs->clear();
  label_data_vecs->resize(max_label >= 0 ? static_cast<size_t>(max_label + 1) : 0);
  for (size_t row = 0; row < data_label_vecs.size(); ++row) {
    for (auto label : data_label_vecs[row]) {
      (*label_data_vecs)[static_cast<size_t>(label)].push_back(static_cast<int>(row));
    }
  }
}

void select_query_subset_by_ids(const std::vector<int64_t>& query_ids,
                                uint32_t dim,
                                std::vector<float>& queries,
                                std::vector<std::vector<int>>& query_label_vecs,
                                uint32_t* query_count_out)
{
  auto original_queries = queries;
  auto original_labels  = query_label_vecs;
  auto original_count   = query_label_vecs.size();

  queries.resize(static_cast<std::size_t>(query_ids.size()) * dim);
  query_label_vecs.clear();
  query_label_vecs.reserve(query_ids.size());

  for (std::size_t i = 0; i < query_ids.size(); ++i) {
    auto query_id = query_ids[i];
    if (query_id < 0 || query_id >= static_cast<int64_t>(original_count)) {
      throw std::runtime_error("Query id out of range in query_id_list_file: " +
                               std::to_string(query_id));
    }
    std::copy_n(original_queries.data() + query_id * dim,
                dim,
                queries.data() + static_cast<int64_t>(i) * dim);
    query_label_vecs.push_back(original_labels[query_id]);
  }

  *query_count_out = static_cast<uint32_t>(query_ids.size());
}

double compute_recall_host(const std::vector<uint32_t>& neighbors,
                           const std::vector<uint32_t>& gt_indices,
                           int64_t n_queries,
                           int topk)
{
  double total_recall = 0.0;
  for (int64_t i = 0; i < n_queries; ++i) {
    int matches = 0;
    auto* nbr = neighbors.data() + i * topk;
    auto* gt = gt_indices.data() + i * topk;
    for (int j = 0; j < topk; ++j) {
      if (nbr[j] == UINT32_MAX) { continue; }
      for (int k = 0; k < topk; ++k) {
        if (nbr[j] == gt[k]) {
          matches++;
          break;
        }
      }
    }
    total_recall += static_cast<double>(matches) / static_cast<double>(topk);
  }
  return total_recall / static_cast<double>(n_queries);
}

struct latency_summary {
  double total_ms = 0.0;
  double avg_ms   = 0.0;
  double p50_ms   = 0.0;
  double p95_ms   = 0.0;
  double max_ms   = 0.0;
};

struct dynamic_run_policy {
  int min_num_runs = 0;
  int max_num_runs = 0;
  int stability_window = 0;
  int stable_subset_size = 0;
  double qps_rel_tol = 0.0;
  double latency_rel_tol = 0.0;
  double trend_guard_rel_tol = 0.0;
};

struct dynamic_stability_result {
  bool triggered = false;
  std::vector<int> subset_run_numbers;
  double subset_mean_qps = 0.0;
  double subset_mean_latency_ms = 0.0;
  double qps_rel_span = 0.0;
  double latency_rel_span = 0.0;
};

double percentile_from_sorted_samples(const std::vector<double>& sorted_samples, double percentile)
{
  if (sorted_samples.empty()) { return 0.0; }

  auto position = percentile * static_cast<double>(sorted_samples.size() - 1);
  auto lower = static_cast<std::size_t>(position);
  auto upper = std::min(lower + 1, sorted_samples.size() - 1);
  auto fraction = position - static_cast<double>(lower);
  return sorted_samples[lower] +
         (sorted_samples[upper] - sorted_samples[lower]) * fraction;
}

latency_summary summarize_latencies_ms(const std::vector<double>& samples_ms)
{
  latency_summary summary;
  if (samples_ms.empty()) { return summary; }

  auto sorted_samples = samples_ms;
  std::sort(sorted_samples.begin(), sorted_samples.end());
  for (auto sample_ms : samples_ms) {
    summary.total_ms += sample_ms;
  }

  summary.avg_ms = summary.total_ms / static_cast<double>(samples_ms.size());
  summary.p50_ms = percentile_from_sorted_samples(sorted_samples, 0.50);
  summary.p95_ms = percentile_from_sorted_samples(sorted_samples, 0.95);
  summary.max_ms = sorted_samples.back();
  return summary;
}

json latency_summary_to_json(const latency_summary& summary)
{
  return json{{"total_ms", summary.total_ms},
              {"avg_ms", summary.avg_ms},
              {"p50_ms", summary.p50_ms},
              {"p95_ms", summary.p95_ms},
              {"max_ms", summary.max_ms}};
}

double relative_span_for_subset(const std::vector<double>& samples,
                                const std::vector<int>& subset_indices,
                                double* mean_out)
{
  if (subset_indices.empty()) {
    if (mean_out != nullptr) { *mean_out = 0.0; }
    return std::numeric_limits<double>::infinity();
  }

  double sum = 0.0;
  double min_value = std::numeric_limits<double>::infinity();
  double max_value = 0.0;
  for (auto index : subset_indices) {
    auto value = samples[static_cast<std::size_t>(index)];
    sum += value;
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  }

  auto mean = sum / static_cast<double>(subset_indices.size());
  if (mean_out != nullptr) { *mean_out = mean; }

  auto denom = std::max(std::fabs(mean), 1e-9);
  return (max_value - min_value) / denom;
}

dynamic_stability_result evaluate_dynamic_stability(
  const std::vector<double>& run_qps_samples,
  const std::vector<double>& run_latency_samples_ms,
  const dynamic_run_policy& policy)
{
  dynamic_stability_result result;
  auto executed_runs = static_cast<int>(run_qps_samples.size());
  if (executed_runs < policy.min_num_runs || executed_runs <= 0) { return result; }

  auto window = std::min(policy.stability_window, executed_runs);
  if (window <= 0 || policy.stable_subset_size <= 0 || policy.stable_subset_size > window) {
    return result;
  }

  auto window_start = executed_runs - window;
  std::vector<int> window_indices;
  window_indices.reserve(static_cast<std::size_t>(window));
  for (int i = 0; i < window; ++i) {
    window_indices.push_back(window_start + i);
  }

  auto latest_qps = run_qps_samples.back();
  auto latest_latency_ms = run_latency_samples_ms.back();
  auto best_score = std::numeric_limits<double>::infinity();

  std::vector<int> current_subset;
  current_subset.reserve(static_cast<std::size_t>(policy.stable_subset_size));

  std::function<void(int, int)> dfs = [&](int start, int remaining) {
    if (remaining == 0) {
      double subset_mean_qps = 0.0;
      double subset_mean_latency_ms = 0.0;
      auto qps_rel_span =
        relative_span_for_subset(run_qps_samples, current_subset, &subset_mean_qps);
      auto latency_rel_span =
        relative_span_for_subset(run_latency_samples_ms, current_subset, &subset_mean_latency_ms);
      if (qps_rel_span > policy.qps_rel_tol || latency_rel_span > policy.latency_rel_tol) {
        return;
      }

      auto qps_guard = std::fabs(latest_qps - subset_mean_qps) /
                       std::max(std::fabs(subset_mean_qps), 1e-9);
      auto latency_guard = std::fabs(latest_latency_ms - subset_mean_latency_ms) /
                           std::max(std::fabs(subset_mean_latency_ms), 1e-9);
      if (qps_guard > policy.trend_guard_rel_tol || latency_guard > policy.trend_guard_rel_tol) {
        return;
      }

      auto score = qps_rel_span + latency_rel_span + qps_guard + latency_guard;
      if (score >= best_score) { return; }

      best_score = score;
      result.triggered = true;
      result.subset_run_numbers.clear();
      result.subset_run_numbers.reserve(current_subset.size());
      for (auto index : current_subset) {
        result.subset_run_numbers.push_back(index + 1);
      }
      result.subset_mean_qps = subset_mean_qps;
      result.subset_mean_latency_ms = subset_mean_latency_ms;
      result.qps_rel_span = qps_rel_span;
      result.latency_rel_span = latency_rel_span;
      return;
    }

    for (int i = start; i <= window - remaining; ++i) {
      current_subset.push_back(window_indices[static_cast<std::size_t>(i)]);
      dfs(i + 1, remaining - 1);
      current_subset.pop_back();
    }
  };

  dfs(0, policy.stable_subset_size);
  return result;
}

struct vecflow_cache_counters_snapshot {
  std::uint64_t phoenix_graph_access_events = 0;
  std::uint64_t phoenix_graph_hbm_hits = 0;
  std::uint64_t phoenix_graph_dram_hits = 0;
  std::uint64_t phoenix_graph_ssd_loads = 0;
  std::uint64_t phoenix_graph_hbm_evictions = 0;
  std::uint64_t phoenix_graph_dram_evictions = 0;

  std::uint64_t phoenix_dataset_hbm_hits = 0;
  std::uint64_t phoenix_dataset_dram_hits = 0;
  std::uint64_t phoenix_dataset_ssd_loads = 0;
  std::uint64_t phoenix_dataset_hbm_evictions = 0;
  std::uint64_t phoenix_dataset_dram_evictions = 0;

  std::uint64_t bfs_access_events = 0;
  std::uint64_t bfs_hbm_hits = 0;
  std::uint64_t bfs_dram_hits = 0;
  std::uint64_t bfs_ssd_loads = 0;
  std::uint64_t bfs_hbm_evictions = 0;
  std::uint64_t bfs_dram_evictions = 0;
};

struct strict_qps_sample {
  double elapsed_seconds = 0.0;
  std::int64_t queries_completed = 0;
  vecflow_cache_counters_snapshot cache_counters{};
};

template <typename data_t>
vecflow_cache_counters_snapshot capture_vecflow_cache_counters(const vecflow::index<data_t>& index)
{
  vecflow_cache_counters_snapshot snapshot;

  if (index.phoenix_label_cache != nullptr) {
    std::lock_guard<std::mutex> lock(index.phoenix_label_cache->mutex);
    snapshot.phoenix_graph_access_events = index.phoenix_label_cache->access_events;
    snapshot.phoenix_graph_hbm_hits = index.phoenix_label_cache->hbm_hits;
    snapshot.phoenix_graph_dram_hits = index.phoenix_label_cache->dram_hits;
    snapshot.phoenix_graph_ssd_loads = index.phoenix_label_cache->ssd_loads;
    snapshot.phoenix_graph_hbm_evictions = index.phoenix_label_cache->hbm_evictions;
    snapshot.phoenix_graph_dram_evictions = index.phoenix_label_cache->dram_evictions;
  }

  if (index.phoenix_label_dataset_cache != nullptr) {
    std::lock_guard<std::mutex> lock(index.phoenix_label_dataset_cache->mutex);
    snapshot.phoenix_dataset_hbm_hits = index.phoenix_label_dataset_cache->hbm_hits;
    snapshot.phoenix_dataset_dram_hits = index.phoenix_label_dataset_cache->dram_hits;
    snapshot.phoenix_dataset_ssd_loads = index.phoenix_label_dataset_cache->ssd_loads;
    snapshot.phoenix_dataset_hbm_evictions = index.phoenix_label_dataset_cache->hbm_evictions;
    snapshot.phoenix_dataset_dram_evictions =
      index.phoenix_label_dataset_cache->dram_evictions;
  }

  if (index.bfs_cache != nullptr) {
    std::scoped_lock lock(index.bfs_cache->hbm_mutex,
                          index.bfs_cache->dram_mutex,
                          index.bfs_cache->access_mutex);
    snapshot.bfs_access_events = index.bfs_cache->access_events;
    snapshot.bfs_hbm_hits = index.bfs_cache->hbm_hits;
    snapshot.bfs_dram_hits = index.bfs_cache->dram_hits;
    snapshot.bfs_ssd_loads = index.bfs_cache->ssd_loads;
    snapshot.bfs_hbm_evictions = index.bfs_cache->hbm_evictions;
    snapshot.bfs_dram_evictions = index.bfs_cache->dram_evictions;
  }

  return snapshot;
}

vecflow_cache_counters_snapshot subtract_vecflow_cache_counters(
  const vecflow_cache_counters_snapshot& after,
  const vecflow_cache_counters_snapshot& before)
{
  vecflow_cache_counters_snapshot delta;
  delta.phoenix_graph_access_events =
    after.phoenix_graph_access_events - before.phoenix_graph_access_events;
  delta.phoenix_graph_hbm_hits = after.phoenix_graph_hbm_hits - before.phoenix_graph_hbm_hits;
  delta.phoenix_graph_dram_hits = after.phoenix_graph_dram_hits - before.phoenix_graph_dram_hits;
  delta.phoenix_graph_ssd_loads = after.phoenix_graph_ssd_loads - before.phoenix_graph_ssd_loads;
  delta.phoenix_graph_hbm_evictions =
    after.phoenix_graph_hbm_evictions - before.phoenix_graph_hbm_evictions;
  delta.phoenix_graph_dram_evictions =
    after.phoenix_graph_dram_evictions - before.phoenix_graph_dram_evictions;

  delta.phoenix_dataset_hbm_hits =
    after.phoenix_dataset_hbm_hits - before.phoenix_dataset_hbm_hits;
  delta.phoenix_dataset_dram_hits =
    after.phoenix_dataset_dram_hits - before.phoenix_dataset_dram_hits;
  delta.phoenix_dataset_ssd_loads =
    after.phoenix_dataset_ssd_loads - before.phoenix_dataset_ssd_loads;
  delta.phoenix_dataset_hbm_evictions =
    after.phoenix_dataset_hbm_evictions - before.phoenix_dataset_hbm_evictions;
  delta.phoenix_dataset_dram_evictions =
    after.phoenix_dataset_dram_evictions - before.phoenix_dataset_dram_evictions;

  delta.bfs_access_events = after.bfs_access_events - before.bfs_access_events;
  delta.bfs_hbm_hits = after.bfs_hbm_hits - before.bfs_hbm_hits;
  delta.bfs_dram_hits = after.bfs_dram_hits - before.bfs_dram_hits;
  delta.bfs_ssd_loads = after.bfs_ssd_loads - before.bfs_ssd_loads;
  delta.bfs_hbm_evictions = after.bfs_hbm_evictions - before.bfs_hbm_evictions;
  delta.bfs_dram_evictions = after.bfs_dram_evictions - before.bfs_dram_evictions;
  return delta;
}

json vecflow_cache_component_to_json(std::uint64_t access_events,
                                     std::uint64_t hbm_hits,
                                     std::uint64_t dram_hits,
                                     std::uint64_t ssd_loads,
                                     std::uint64_t hbm_evictions,
                                     std::uint64_t dram_evictions)
{
  return json{{"access_events", access_events},
              {"hbm_hits", hbm_hits},
              {"dram_hits", dram_hits},
              {"ssd_loads", ssd_loads},
              {"hbm_evictions", hbm_evictions},
              {"dram_evictions", dram_evictions}};
}

json vecflow_cache_counters_to_json(const vecflow_cache_counters_snapshot& counters)
{
  return json{
    {"phoenix_graph",
     vecflow_cache_component_to_json(counters.phoenix_graph_access_events,
                                     counters.phoenix_graph_hbm_hits,
                                     counters.phoenix_graph_dram_hits,
                                     counters.phoenix_graph_ssd_loads,
                                     counters.phoenix_graph_hbm_evictions,
                                     counters.phoenix_graph_dram_evictions)},
    {"phoenix_dataset",
     vecflow_cache_component_to_json(0,
                                     counters.phoenix_dataset_hbm_hits,
                                     counters.phoenix_dataset_dram_hits,
                                     counters.phoenix_dataset_ssd_loads,
                                     counters.phoenix_dataset_hbm_evictions,
                                     counters.phoenix_dataset_dram_evictions)},
    {"tiered_bfs",
     vecflow_cache_component_to_json(counters.bfs_access_events,
                                     counters.bfs_hbm_hits,
                                     counters.bfs_dram_hits,
                                     counters.bfs_ssd_loads,
                                     counters.bfs_hbm_evictions,
                                     counters.bfs_dram_evictions)}};
}

json strict_qps_sample_to_json(const strict_qps_sample& sample)
{
  return json{{"elapsed_seconds", sample.elapsed_seconds},
              {"queries_completed", sample.queries_completed},
              {"cache_counters", vecflow_cache_counters_to_json(sample.cache_counters)}};
}

json storage_tier_stats_to_json(const vecflow::storage_tier_stats& stats)
{
  return json{{"labels", stats.labels}, {"bytes", stats.bytes}};
}

json storage_component_stats_to_json(const vecflow::storage_component_stats& stats)
{
  return json{{"hbm", storage_tier_stats_to_json(stats.hbm)},
              {"dram", storage_tier_stats_to_json(stats.dram)},
              {"ssd", storage_tier_stats_to_json(stats.ssd)},
              {"resident_hbm", storage_tier_stats_to_json(stats.resident_hbm)},
              {"resident_dram", storage_tier_stats_to_json(stats.resident_dram)},
              {"total_labels", stats.total_labels},
              {"total_bytes", stats.total_bytes}};
}

json storage_stats_info_to_json(const vecflow::storage_stats_info& stats)
{
  return json{{"graph", storage_component_stats_to_json(stats.graph)},
              {"dataset", storage_component_stats_to_json(stats.dataset)},
              {"bfs", storage_component_stats_to_json(stats.bfs)},
              {"access_events", stats.access_events}};
}

std::vector<vecflow_mg_device_context> build_vecflow_mg_contexts(
  const std::vector<int>& device_ids,
  const std::vector<float>& h_data,
  uint32_t n_rows,
  uint32_t dim,
  const std::vector<std::vector<int>>& data_label_vecs,
  int graph_degree,
  int specificity_threshold,
  const std::string& graph_fname,
  const std::string& bfs_fname,
  bool force_rebuild,
  vecflow::graph_builder_type builder,
  int tagore_iterations)
{
  std::vector<vecflow_mg_device_context> contexts;
  contexts.resize(device_ids.size());

  for (size_t i = 0; i < device_ids.size(); ++i) {
    auto device_id = device_ids[i];
    RAFT_CUDA_TRY(cudaSetDevice(device_id));
    auto& ctx = contexts[i];
    ctx.device_id = device_id;
    shared_resources::configured_raft_resources res;
    ctx.dataset.emplace(raft::make_device_matrix<float, int64_t>(res, n_rows, dim));
    raft::copy(ctx.dataset->data_handle(),
               h_data.data(),
               static_cast<int64_t>(n_rows) * dim,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    ctx.index.emplace(vecflow::build(res,
                                     raft::make_const_mdspan(ctx.dataset->view()),
                                     data_label_vecs,
                                     graph_degree,
                                     specificity_threshold,
                                     graph_fname,
                                     bfs_fname,
                                     force_rebuild,
                                     builder,
                                     tagore_iterations));
    raft::resource::sync_stream(res);
  }

  return contexts;
}

void search_vecflow_mg_chunk(vecflow_mg_device_context& ctx,
                             int worker_id,
                             int worker_count,
                             const std::vector<float>& h_queries,
                             const std::vector<uint32_t>& h_query_labels,
                             uint32_t dim,
                             int64_t query_offset,
                             int64_t query_count,
                             int itopk_size,
                             int topk,
                             std::vector<uint32_t>& h_neighbors_out,
                             std::vector<float>& h_distances_out)
{
  if (query_count == 0) { return; }

  RAFT_CUDA_TRY(cudaSetDevice(ctx.device_id));
  shared_resources::thread_id = worker_id;
  shared_resources::n_threads = worker_count;

  shared_resources::configured_raft_resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  auto d_queries = raft::make_device_matrix<float, int64_t>(res, query_count, dim);
  auto d_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, query_count);
  auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, query_count, topk);
  auto d_distances = raft::make_device_matrix<float, int64_t>(res, query_count, topk);

  raft::copy(d_queries.data_handle(),
             h_queries.data() + query_offset * dim,
             query_count * dim,
             stream);
  raft::copy(d_query_labels.data_handle(),
             h_query_labels.data() + query_offset,
             query_count,
             stream);

  vecflow::search(res,
                  *ctx.index,
                  raft::make_const_mdspan(d_queries.view()),
                  d_query_labels.view(),
                  itopk_size,
                  d_neighbors.view(),
                  d_distances.view());

  raft::copy(h_neighbors_out.data() + query_offset * topk,
             d_neighbors.data_handle(),
             query_count * topk,
             stream);
  raft::copy(h_distances_out.data() + query_offset * topk,
             d_distances.data_handle(),
             query_count * topk,
             stream);
  raft::resource::sync_stream(res);
}

void build_cagra_index(shared_resources::configured_raft_resources& dev_resources,
                       cagra::index<float, uint32_t>& index,
                       const raft::device_matrix_view<const float, int64_t>& dataset,
                       const std::string& index_path,
                       int graph_degree) {

  if (std::filesystem::exists(index_path)) {
    std::cout << "Loading existing CAGRA index from " << index_path << std::endl;
    cagra::deserialize(dev_resources, index_path, &index);
    index.update_dataset(dev_resources, dataset);
  } else {
    std::cout << "Building new CAGRA index..." << std::endl;
    cagra::index_params index_params;
    index_params.intermediate_graph_degree = graph_degree * 2;
    index_params.graph_degree = graph_degree;

    index = cagra::build(dev_resources, index_params, dataset);

    std::cout << "Saving CAGRA index to " << index_path << std::endl;
    cagra::serialize(dev_resources, index_path, index, false);
  }

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;
}

__global__ void filter_neighbors_kernel(const uint32_t* neighbors,
                                        const uint32_t* query_labels,
                                        const uint32_t* label_bits,
                                        uint32_t* filtered_neighbors,
                                        int64_t n_queries,
                                        int64_t itopk_size,
                                        int64_t topk,
                                        int num_labels,
                                        int64_t n_vectors) {

  int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_idx >= n_queries) return;

  constexpr size_t BITS_PER_WORD = 32;
  size_t NUM_WORDS = (num_labels + BITS_PER_WORD - 1) / BITS_PER_WORD;

  uint32_t query_label = query_labels[query_idx];
  int filtered_count = 0;

  if (query_label >= num_labels) {
     while (filtered_count < topk) {
       filtered_neighbors[query_idx * topk + filtered_count] = UINT32_MAX;
       filtered_count++;
     }
     return;
  }

  size_t query_word_idx = query_label / BITS_PER_WORD;
  size_t query_bit_idx = query_label % BITS_PER_WORD;
  uint32_t query_bit_mask = 1u << query_bit_idx;

  int match_count = 0;
  for (int j = 0; j < itopk_size && filtered_count < topk; ++j) {
    uint32_t neighbor_idx = neighbors[query_idx * itopk_size + j];

    if (neighbor_idx == UINT32_MAX || neighbor_idx >= n_vectors) continue;

    size_t bit_array_idx = neighbor_idx * NUM_WORDS + query_word_idx;

    // Check bounds ONLY to prevent crash, actual data might still be wrong
    if (bit_array_idx >= (size_t)(n_vectors * NUM_WORDS)) continue;

    uint32_t word = label_bits[bit_array_idx];

    if ((word & query_bit_mask) != 0) {
      filtered_neighbors[query_idx * topk + filtered_count] = neighbor_idx;
      filtered_count++;
      match_count++;
    }
  }

  // Fill remaining slots with UINT32_MAX
  while (filtered_count < topk) {
    filtered_neighbors[query_idx * topk + filtered_count] = UINT32_MAX;
    filtered_count++;
  }
}

template<typename index_t = int, typename bitmap_t = uint32_t>
__global__ void create_bitmap_filter_kernel(const index_t* __restrict__ row_offsets,
                                            const index_t* __restrict__ indices,
                                            const uint32_t* __restrict__ query_labels,
                                            bitmap_t* __restrict__ bitmap,
                                            const index_t num_queries,
                                            const index_t num_cols,
                                            const index_t words_per_row) {

  const index_t query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  // Get start and end indices for this query's label
  // Note: Assuming label_data_vecs was used to build row_offsets and indices correctly
  // such that query_labels[query_idx] maps to the correct row in that structure.
  const index_t label_row = query_labels[query_idx];
  // Add bounds check for label_row if needed, depending on how row_offsets is sized
  // if (label_row >= num_unique_labels) return; // Example check

  const index_t start = row_offsets[label_row];
  const index_t end = row_offsets[label_row + 1];

  // Each thread handles one index in the label list
  for (index_t i = threadIdx.x; i < (end - start); i += blockDim.x) {
    const index_t idx = indices[start + i]; // This is the vector_idx that has the label
    if (idx >= num_cols) continue;  // Check against n_database

    // Calculate position in bitmap for this query and this vector_idx
    const index_t word_idx = query_idx * words_per_row + (idx / (sizeof(bitmap_t) * 8));
    const unsigned bit_offset = idx % (sizeof(bitmap_t) * 8);

    // Set bit using atomic operation
    atomicOr(&bitmap[word_idx], bitmap_t(1) << bit_offset);
  }
}

void create_bitmap_filter_fast(raft::resources const& handle,
                               const std::vector<std::vector<int>>& label_data_vecs,
                               const raft::device_vector_view<const uint32_t, int64_t>& query_labels_view,
                               int64_t n_queries,
                               int64_t n_database,
                               const raft::device_matrix_view<uint32_t, int64_t>& bitmap) {

  // Calculate total size needed for indices based on label_data_vecs
  int64_t total_indices = 0;
  for (const auto& vec : label_data_vecs) {
    total_indices += vec.size();
  }

  // Create and fill row offsets on host
  std::vector<int64_t> h_row_offsets(label_data_vecs.size() + 1, 0);
  int64_t current_offset = 0;
  for (size_t i = 0; i < label_data_vecs.size(); i++) {
    h_row_offsets[i] = current_offset;
    current_offset += label_data_vecs[i].size();
  }
  h_row_offsets[label_data_vecs.size()] = current_offset;

  // Create indices array on host (vector IDs for each label)
  std::vector<int64_t> h_indices(total_indices);
  current_offset = 0;
  for (const auto& vec : label_data_vecs) {
    // Important: Convert int to int64_t if needed, ensure types match d_indices
    std::transform(vec.begin(), vec.end(), h_indices.begin() + current_offset,
                   [](int val){ return static_cast<int64_t>(val); });
    current_offset += vec.size();
  }

  // Copy CSR structure data to GPU
  auto d_row_offsets = raft::make_device_vector<int64_t, int64_t>(handle, h_row_offsets.size());
  auto d_indices = raft::make_device_vector<int64_t, int64_t>(handle, h_indices.size());

  raft::update_device(d_row_offsets.data_handle(),
                      h_row_offsets.data(),
                      h_row_offsets.size(),
                      raft::resource::get_cuda_stream(handle));

  raft::update_device(d_indices.data_handle(),
                      h_indices.data(),
                      h_indices.size(),
                      raft::resource::get_cuda_stream(handle));

  // Get the actual words_per_row from the bitmap view
  const int64_t words_per_row = bitmap.extent(1);

  // Zero initialize bitmap
  RAFT_CUDA_TRY(cudaMemsetAsync(
      bitmap.data_handle(),
      0,
      n_queries * words_per_row * sizeof(uint32_t),
      raft::resource::get_cuda_stream(handle)));

  // Launch kernel
  const int block_size = 256;
  // GridDim is n_queries - each block handles one query
  create_bitmap_filter_kernel<<<n_queries, block_size, 0, raft::resource::get_cuda_stream(handle)>>>(
    d_row_offsets.data_handle(),
    d_indices.data_handle(),
    query_labels_view.data_handle(), // Pass device pointer
    bitmap.data_handle(),            // Pass device pointer
    n_queries,
    n_database,  // This is num_cols in the kernel
    words_per_row
  );

  raft::resource::sync_stream(handle); // Wait for kernel completion
}

// Function for CAGRA search with inline filtering
double cagra_search_inline_filtering(shared_resources::configured_raft_resources& dev_resources,
                                     cagra::index<float, uint32_t>& cagra_index,
                                     const raft::device_matrix_view<const float, int64_t>& queries,
                                     const raft::device_vector_view<const uint32_t, int64_t>& query_labels,
                                     const std::vector<std::vector<int>>& label_data_vecs,
                                     int itopk_size,
                                     int topk,
                                     int num_runs,
                                     int warmup_runs,
                                     raft::device_matrix_view<uint32_t, int64_t> filtered_neighbors) {

  int64_t n_queries = queries.extent(0);
  int64_t n_database = cagra_index.size();

  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;

  auto cagra_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, topk);

  // Create bitmap for this iteration
  const int64_t bits_per_uint32 = sizeof(uint32_t) * 8;
  const int64_t words_per_row = (n_database + bits_per_uint32 - 1) / bits_per_uint32;
  auto bitmap = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_queries, words_per_row);
  create_bitmap_filter_fast(dev_resources,
                            label_data_vecs,
                            query_labels,
                            n_queries,
                            n_database,
                            bitmap.view());
  auto bitmap_view = raft::core::bitmap_view<const uint32_t, int64_t>(bitmap.data_handle(), n_queries, n_database);
  auto filter = cuvs::neighbors::filtering::bitmap_filter<const uint32_t, int64_t>(bitmap_view);

  // --- Warmup Section ---
  for (int i = 0; i < warmup_runs; i++) {
    cagra::search(dev_resources,
                  search_params,
                  cagra_index,
                  queries,
                  filtered_neighbors,
                  cagra_distances.view(),
                  filter);
  }
  raft::resource::sync_stream(dev_resources);

  // --- Timed Section ---
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_runs; i++) {
    cagra::search(dev_resources,
                  search_params,
                  cagra_index,
                  queries,
                  filtered_neighbors,
                  cagra_distances.view(),
                  filter);
  }
  raft::resource::sync_stream(dev_resources);
  auto end_time = std::chrono::high_resolution_clock::now();

  auto total_time = std::chrono::duration<double>(end_time - start_time).count();
  double qps = num_runs * n_queries / total_time;

  return qps;
}

double cagra_search_with_post_processing(shared_resources::configured_raft_resources& dev_resources,
                                         cagra::index<float, uint32_t>& cagra_index,
                                         const raft::device_matrix_view<const float, int64_t>& queries,
                                         const raft::device_vector_view<const uint32_t, int64_t>& query_labels,
                                         const std::vector<std::vector<int>>& label_data_vecs,
                                         int num_labels,
                                         int itopk_size,
                                         int topk,
                                         int num_runs,
                                         int warmup_runs,
                                         raft::device_matrix_view<uint32_t, int64_t> filtered_neighbors) {

  auto stream = raft::resource::get_cuda_stream(dev_resources);
  int64_t n_queries = queries.extent(0);
  int64_t n_vectors = cagra_index.size();  // Use actual index size

  auto cagra_neighbors = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, n_queries, itopk_size);
  auto cagra_distances = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, itopk_size);
  auto d_query_labels = raft::make_device_vector<uint32_t>(dev_resources, n_queries);
  raft::copy(d_query_labels.data_handle(), query_labels.data_handle(), n_queries, stream);

  constexpr size_t BITS_PER_WORD = 32;
  size_t NUM_WORDS = (num_labels + BITS_PER_WORD - 1) / BITS_PER_WORD;
  size_t total_label_words = (size_t)n_vectors * NUM_WORDS; // Use size_t

  std::vector<uint32_t> h_label_bits(total_label_words, 0);

  size_t max_vectors_for_labels = std::min(static_cast<size_t>(n_vectors), label_data_vecs.size());
  for (int label = 0; label < num_labels; label++) {
    if (label >= label_data_vecs.size()) continue; // Skip if label index is out of bounds for label_data_vecs
    const auto& vectors_with_this_label = label_data_vecs[label];
    size_t word_idx = label / BITS_PER_WORD;
    size_t bit_idx = label % BITS_PER_WORD;
    uint32_t bit_mask = 1u << bit_idx;
    for (int vector_idx : vectors_with_this_label) {
      if (vector_idx >= 0 && vector_idx < n_vectors) { // Check against n_vectors
        size_t array_idx = (size_t)vector_idx * NUM_WORDS + word_idx; // Cast vector_idx
        if (array_idx < h_label_bits.size()) {
          h_label_bits[array_idx] |= bit_mask;
        }
      }
    }
  }

  auto d_label_bits = raft::make_device_vector<uint32_t>(dev_resources, total_label_words);

  raft::copy(d_label_bits.data_handle(),
             h_label_bits.data(),
             h_label_bits.size(),
             stream);
  // Ensure copy is complete before kernel launch
  raft::resource::sync_stream(dev_resources);

  cagra::search_params search_params;
  search_params.itopk_size = itopk_size;

  // Warmup runs
  for (int i = 0; i < warmup_runs; i++) {
    cagra::search(dev_resources, search_params, cagra_index, queries,
                  cagra_neighbors.view(), cagra_distances.view());
    int block_size = 256;
    int grid_size = (n_queries + block_size - 1) / block_size;
    filter_neighbors_kernel<<<grid_size, block_size, 0, stream>>>(
      cagra_neighbors.data_handle(),
      d_query_labels.data_handle(),
      d_label_bits.data_handle(),
      filtered_neighbors.data_handle(),
      n_queries,
      itopk_size,
      topk,
      num_labels,
      n_vectors
    );
    raft::resource::sync_stream(dev_resources);
  }
  raft::resource::sync_stream(dev_resources);

  // Timed runs
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_runs; i++) {
    cagra::search(dev_resources, search_params, cagra_index, queries,
                  cagra_neighbors.view(), cagra_distances.view());
    int block_size = 256;
    int grid_size = (n_queries + block_size - 1) / block_size;
    filter_neighbors_kernel<<<grid_size, block_size, 0, stream>>>(
      cagra_neighbors.data_handle(),
      d_query_labels.data_handle(),
      d_label_bits.data_handle(),
      filtered_neighbors.data_handle(),
      n_queries,
      itopk_size,
      topk,
      num_labels,
      n_vectors
    );
    raft::resource::sync_stream(dev_resources);
  }
  raft::resource::sync_stream(dev_resources);
  auto end_time = std::chrono::high_resolution_clock::now();

  auto total_time = std::chrono::duration<double>(end_time - start_time).count();
  double qps = num_runs * n_queries / total_time;

  return qps;
}


int main(int argc, char** argv) {
	// Check if config file is provided
	std::string config_file;
	if (argc < 3 || std::string(argv[1]) != "--config") {
		printf("No config file provided. Using default configuration file './config/default_config.json'.\n");
		config_file = "../src/bench/config.json";
	} else {
		config_file = argv[2];
	}

	// Variables to store configuration
	std::string data_dir;
	std::string data_fname;
	std::string query_fname;
	std::string data_label_fname;
	std::string query_label_fname;
	std::string ivf_graph_fname;
	std::string ivf_graph_tagore_fname;
	std::string ivf_bfs_fname;
	std::string cagra_index_fname;
	std::string ground_truth_fname;
	std::vector<int> itopk_sizes;
	int specificity_threshold;
	int graph_degree;
	int topk;
	int num_runs;
	int warmup_runs;
	int min_num_runs;
	int max_num_runs;
	int stability_window;
	int stable_subset_size;
	double qps_stability_rel_tol;
	double latency_stability_rel_tol;
	double trend_guard_rel_tol;
	bool force_rebuild = false;
	int tagore_iterations = 10;
  bool use_phoenix_graph_load = false;
  bool use_phoenix_label_load = false;
  std::uint64_t phoenix_label_cache_bytes = 1ULL << 30;
  std::uint64_t phoenix_label_dram_cache_bytes = 0;
  std::uint64_t phoenix_label_prefetch_max_bytes = 0;
  std::uint64_t phoenix_label_dataset_cache_bytes = 1ULL << 30;
  std::uint64_t phoenix_label_dataset_dram_cache_bytes = 0;
  std::uint64_t phoenix_label_dataset_prefetch_max_bytes = 0;
  std::uint64_t phoenix_label_rebalance_interval_queries = 64;
  bool enable_bfs_tiered_cache = false;
  std::uint64_t bfs_hbm_cache_bytes = 0;
  std::uint64_t bfs_dram_cache_bytes = 0;
  std::uint64_t bfs_prefetch_max_bytes = 0;
  std::uint64_t bfs_rebalance_interval_queries = 64;
  bool cascade_eviction = true;
	std::vector<std::string> algorithms_to_run;
  std::vector<int> device_ids{0};
  int64_t query_offset = 0;
  int64_t query_count = -1;
  std::string query_id_list_file;
  std::string allowed_labels_file;
  std::string query_label_mode = "single";
  bool skip_recall = false;
  bool strict_qps_sampling_enabled = false;
  double strict_qps_sampling_interval_seconds = 1.0;
	std::string output_json_file;

	// Load configuration from file
	std::ifstream file(config_file);
	if (!file.is_open()) {
		fprintf(stderr, "Unable to open config file: %s\n", config_file.c_str());
		return 1;
	}

	try {
		printf("Loading configuration from %s\n", config_file.c_str());
		json config;
		file >> config;

		// Load all parameters directly from config
		data_dir = config["data_dir"];
		data_fname = config["data_fname"];
		query_fname = config["query_fname"];
		data_label_fname = config["data_label_fname"];
		query_label_fname = config["query_label_fname"];
		itopk_sizes = config["itopk_size"].get<std::vector<int>>();
		specificity_threshold = config["spec_threshold"];
		graph_degree = config["graph_degree"];
		topk = config["topk"];
		num_runs = config["num_runs"];
		warmup_runs = config["warmup_runs"];
		min_num_runs = config.value("min_num_runs", num_runs);
		max_num_runs = config.value("max_num_runs", num_runs);
		stability_window = config.value("stability_window", max_num_runs);
		stable_subset_size = config.value("stable_subset_size", min_num_runs);
		qps_stability_rel_tol = config.value("qps_stability_rel_tol", 0.0);
		latency_stability_rel_tol = config.value("latency_stability_rel_tol", 0.0);
		trend_guard_rel_tol = config.value("trend_guard_rel_tol", qps_stability_rel_tol);
		force_rebuild = config["force_rebuild"];

		ivf_graph_fname = config["ivf_graph_fname"];
		ivf_graph_tagore_fname = config.value("ivf_graph_tagore_fname", "ivf_graph_tagore.bin");
		ivf_bfs_fname = config["ivf_bfs_fname"];
		cagra_index_fname = config["cagra_index_fname"];
		ground_truth_fname = config["ground_truth_fname"];
		tagore_iterations = config.value("tagore_iterations", 10);
    use_phoenix_graph_load = config.value("use_phoenix_graph_load", false);
    use_phoenix_label_load = config.value("use_phoenix_label_load", false);
    phoenix_label_cache_bytes =
      config.value("phoenix_label_cache_bytes", static_cast<std::uint64_t>(1ULL << 30));
    phoenix_label_dram_cache_bytes =
      config.value("phoenix_label_dram_cache_bytes", static_cast<std::uint64_t>(0));
    phoenix_label_prefetch_max_bytes =
      config.value("phoenix_label_prefetch_max_bytes", static_cast<std::uint64_t>(0));
    phoenix_label_dataset_cache_bytes = config.value(
      "phoenix_label_dataset_cache_bytes",
      static_cast<std::uint64_t>(phoenix_label_cache_bytes));
    phoenix_label_dataset_dram_cache_bytes = config.value(
      "phoenix_label_dataset_dram_cache_bytes",
      static_cast<std::uint64_t>(phoenix_label_dram_cache_bytes));
    phoenix_label_dataset_prefetch_max_bytes = config.value(
      "phoenix_label_dataset_prefetch_max_bytes",
      static_cast<std::uint64_t>(phoenix_label_prefetch_max_bytes));
    phoenix_label_rebalance_interval_queries = config.value(
      "phoenix_label_rebalance_interval_queries", static_cast<std::uint64_t>(64));
    enable_bfs_tiered_cache = config.contains("bfs_hbm_cache_bytes") ||
                              config.contains("bfs_dram_cache_bytes") ||
                              config.contains("bfs_prefetch_max_bytes") ||
                              config.contains("bfs_rebalance_interval_queries");
    bfs_hbm_cache_bytes =
      config.value("bfs_hbm_cache_bytes", static_cast<std::uint64_t>(0));
    bfs_dram_cache_bytes =
      config.value("bfs_dram_cache_bytes", static_cast<std::uint64_t>(0));
    bfs_prefetch_max_bytes =
      config.value("bfs_prefetch_max_bytes", static_cast<std::uint64_t>(0));
    bfs_rebalance_interval_queries = config.value(
      "bfs_rebalance_interval_queries", static_cast<std::uint64_t>(64));
    cascade_eviction = config.value("cascade_eviction", true);

		// Load new parameters
		algorithms_to_run = config["algorithms_to_run"].get<std::vector<std::string>>();
		device_ids = config.value("device_ids", std::vector<int>{0});
		query_offset = config.value("query_offset", static_cast<int64_t>(0));
		query_count = config.value("query_count", static_cast<int64_t>(-1));
		query_id_list_file = config.value("query_id_list_file", std::string{});
		allowed_labels_file = config.value("allowed_labels_file", std::string{});
		query_label_mode = config.value("query_label_mode", std::string{"single"});
    skip_recall = config.value("skip_recall", false);
    strict_qps_sampling_enabled = config.value("strict_qps_sampling_enabled", false);
    strict_qps_sampling_interval_seconds =
      config.value("strict_qps_sampling_interval_seconds", 1.0);
		output_json_file = config["output_json_file"];

	} catch (const std::exception& e) {
		fprintf(stderr, "Error parsing JSON config file: %s\n", e.what());
		return 1;
	}

	// Check if itopk_sizes is empty
	if (itopk_sizes.empty()) {
		fprintf(stderr, "Error: 'itopk_size' array in config file is empty.\n");
		return 1;
	}
	if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "vecflow_mg") !=
	      algorithms_to_run.end() ||
	    std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "vecflow_tagore_mg") !=
	      algorithms_to_run.end()) {
    fprintf(stderr,
            "Error: multi-GPU algorithms are handled by VECFLOW_MG_BENCH. "
            "Run that binary with the same config instead of VECFLOW_BENCH.\n");
    return 1;
  }

  bool use_multi_label_search = false;
  auto multi_label_combine_mode = vecflow::multi_label_query_desc::combine_mode::OR;
  auto multi_label_and_mode = vecflow::multi_label_query_desc::and_strategy::GREEDY;
  auto ground_truth_label_mode = query_label_match_mode::ANY;
  if (query_label_mode == "single") {
    use_multi_label_search = false;
  } else if (query_label_mode == "multi_or") {
    use_multi_label_search = true;
    multi_label_combine_mode = vecflow::multi_label_query_desc::combine_mode::OR;
    ground_truth_label_mode = query_label_match_mode::ANY;
  } else if (query_label_mode == "multi_and_greedy") {
    use_multi_label_search = true;
    multi_label_combine_mode = vecflow::multi_label_query_desc::combine_mode::AND;
    multi_label_and_mode = vecflow::multi_label_query_desc::and_strategy::GREEDY;
    ground_truth_label_mode = query_label_match_mode::ALL;
  } else if (query_label_mode == "multi_and_parallel") {
    use_multi_label_search = true;
    multi_label_combine_mode = vecflow::multi_label_query_desc::combine_mode::AND;
    multi_label_and_mode = vecflow::multi_label_query_desc::and_strategy::PARALLEL;
    ground_truth_label_mode = query_label_match_mode::ALL;
  } else {
    fprintf(stderr, "Error: unsupported query_label_mode '%s'.\n", query_label_mode.c_str());
    return 1;
  }

  if (use_multi_label_search) {
    for (auto const& algorithm : algorithms_to_run) {
      if (algorithm != "vecflow" && algorithm != "vecflow_tagore") {
        fprintf(stderr,
                "Error: query_label_mode=%s currently supports only vecflow/vecflow_tagore.\n",
                query_label_mode.c_str());
        return 1;
      }
    }
  }
	// Sort itopk_sizes for potentially clearer output, though not strictly necessary
	std::sort(itopk_sizes.begin(), itopk_sizes.end());

	// Construct full file paths
	std::string full_data_fname = data_dir + data_fname;
	std::string full_query_fname = data_dir + query_fname;
	std::string full_data_label_fname = data_dir + data_label_fname;
	std::string full_query_label_fname = data_dir + query_label_fname;
	std::string full_ivf_graph_fname = data_dir + ivf_graph_fname;
	std::string full_ivf_graph_tagore_fname = data_dir + ivf_graph_tagore_fname;
	std::string full_ivf_bfs_fname = data_dir + ivf_bfs_fname;
	std::string full_cagra_index_fname = data_dir + cagra_index_fname;
	std::string full_ground_truth_fname = data_dir + ground_truth_fname;
  if (use_multi_label_search) {
    full_ground_truth_fname = append_suffix_to_filename(
      full_ground_truth_fname,
      ground_truth_label_mode == query_label_match_mode::ALL ? "_labelmatch_all"
                                                             : "_labelmatch_any");
  }

	// Print configuration
	printf("\n=== Configuration ===\n");
	printf("iTopK sizes: [ ");
	for(int itopk : itopk_sizes) { printf("%d ", itopk); }
	printf("]\n");
	printf("Specificity threshold: %d\n", specificity_threshold);
	printf("Graph degree: %d\n", graph_degree);
	printf("TopK: %d\n", topk);
	printf("Number of runs: %d\n", num_runs);
	printf("Warmup runs: %d\n", warmup_runs);
	printf("Min benchmark runs: %d\n", min_num_runs);
	printf("Max benchmark runs: %d\n", max_num_runs);
	printf("Stability window: %d\n", stability_window);
	printf("Stable subset size: %d\n", stable_subset_size);
	printf("QPS stability relative tolerance: %.4f\n", qps_stability_rel_tol);
	printf("Latency stability relative tolerance: %.4f\n", latency_stability_rel_tol);
	printf("Trend guard relative tolerance: %.4f\n", trend_guard_rel_tol);
	printf("Tagore iterations: %d\n", tagore_iterations);
  printf("Use Phoenix graph load: %s\n", use_phoenix_graph_load ? "true" : "false");
  printf("Use Phoenix label load: %s\n", use_phoenix_label_load ? "true" : "false");
  printf("Phoenix label cache bytes: %llu\n",
         static_cast<unsigned long long>(phoenix_label_cache_bytes));
  printf("Phoenix label DRAM cache bytes: %llu\n",
         static_cast<unsigned long long>(phoenix_label_dram_cache_bytes));
  printf("Phoenix label prefetch max bytes: %llu\n",
         static_cast<unsigned long long>(phoenix_label_prefetch_max_bytes));
  printf("Phoenix label dataset cache bytes: %llu\n",
         static_cast<unsigned long long>(phoenix_label_dataset_cache_bytes));
  printf("Phoenix label dataset DRAM cache bytes: %llu\n",
         static_cast<unsigned long long>(phoenix_label_dataset_dram_cache_bytes));
  printf("Phoenix label dataset prefetch max bytes: %llu\n",
         static_cast<unsigned long long>(phoenix_label_dataset_prefetch_max_bytes));
  printf("Phoenix label rebalance interval queries: %llu\n",
         static_cast<unsigned long long>(phoenix_label_rebalance_interval_queries));
  printf("Enable BFS tiered cache: %s\n", enable_bfs_tiered_cache ? "true" : "false");
  printf("BFS HBM cache bytes: %llu\n",
         static_cast<unsigned long long>(bfs_hbm_cache_bytes));
  printf("BFS DRAM cache bytes: %llu\n",
         static_cast<unsigned long long>(bfs_dram_cache_bytes));
  printf("BFS prefetch max bytes: %llu\n",
         static_cast<unsigned long long>(bfs_prefetch_max_bytes));
  printf("BFS rebalance interval queries: %llu\n",
         static_cast<unsigned long long>(bfs_rebalance_interval_queries));
  printf("Cascade eviction: %s\n", cascade_eviction ? "true" : "false");
  printf("Device IDs: [ ");
  for (auto device_id : device_ids) { printf("%d ", device_id); }
  printf("]\n");
  if (!query_id_list_file.empty()) {
    printf("Query id list file: %s\n", query_id_list_file.c_str());
  }
  if (!allowed_labels_file.empty()) {
    printf("Allowed labels file: %s\n", allowed_labels_file.c_str());
  }
  printf("Query label mode: %s\n", query_label_mode.c_str());
	printf("Strict QPS sampling enabled: %s\n",
	       strict_qps_sampling_enabled ? "true" : "false");
	printf("Strict QPS sampling interval seconds: %.3f\n",
	       strict_qps_sampling_interval_seconds);
	printf("Algorithms to run: [ ");
	for(const auto& algo : algorithms_to_run) { printf("%s ", algo.c_str()); }
	printf("]\n");
	printf("Output JSON file: %s\n", output_json_file.c_str());

  if (strict_qps_sampling_interval_seconds <= 0.0) {
    fprintf(stderr, "Error: strict_qps_sampling_interval_seconds must be > 0.\n");
    return 1;
  }

  if (min_num_runs <= 0) {
    fprintf(stderr, "Error: min_num_runs must be positive.\n");
    return 1;
  }
  if (max_num_runs < min_num_runs) {
    fprintf(stderr, "Error: max_num_runs must be >= min_num_runs.\n");
    return 1;
  }
  if (num_runs != max_num_runs) {
    printf("Overriding legacy num_runs=%d with max_num_runs=%d for execution control.\n",
           num_runs,
           max_num_runs);
  }
  num_runs = max_num_runs;
  if (stability_window < stable_subset_size) { stability_window = stable_subset_size; }
  if (stability_window < min_num_runs) { stability_window = min_num_runs; }

  dynamic_run_policy dynamic_policy;
  dynamic_policy.min_num_runs = min_num_runs;
  dynamic_policy.max_num_runs = max_num_runs;
  dynamic_policy.stability_window = stability_window;
  dynamic_policy.stable_subset_size = stable_subset_size;
  dynamic_policy.qps_rel_tol = qps_stability_rel_tol;
  dynamic_policy.latency_rel_tol = latency_stability_rel_tol;
  dynamic_policy.trend_guard_rel_tol = trend_guard_rel_tol;

  vecflow::runtime_config runtime_config;
  runtime_config.cascade_eviction      = cascade_eviction;
  runtime_config.enable_bfs_tiered_cache = enable_bfs_tiered_cache;
  runtime_config.use_phoenix_label_load = use_phoenix_label_load;
  runtime_config.use_phoenix_graph_load = use_phoenix_label_load ? false : use_phoenix_graph_load;

  if (enable_bfs_tiered_cache) {
    runtime_config.bfs_hbm_cache_bytes             = static_cast<std::size_t>(bfs_hbm_cache_bytes);
    runtime_config.bfs_dram_cache_bytes            = static_cast<std::size_t>(bfs_dram_cache_bytes);
    runtime_config.bfs_prefetch_max_bytes          = static_cast<std::size_t>(bfs_prefetch_max_bytes);
    runtime_config.bfs_rebalance_interval_queries  =
      static_cast<std::size_t>(bfs_rebalance_interval_queries);
  }

  if (use_phoenix_label_load) {
    runtime_config.phoenix_label_cache_bytes =
      static_cast<std::size_t>(phoenix_label_cache_bytes);
    runtime_config.phoenix_label_dram_cache_bytes =
      static_cast<std::size_t>(phoenix_label_dram_cache_bytes);
    runtime_config.phoenix_label_prefetch_max_bytes =
      static_cast<std::size_t>(phoenix_label_prefetch_max_bytes);
    runtime_config.phoenix_label_dataset_cache_bytes =
      static_cast<std::size_t>(phoenix_label_dataset_cache_bytes);
    runtime_config.phoenix_label_dataset_dram_cache_bytes =
      static_cast<std::size_t>(phoenix_label_dataset_dram_cache_bytes);
    runtime_config.phoenix_label_dataset_prefetch_max_bytes =
      static_cast<std::size_t>(phoenix_label_dataset_prefetch_max_bytes);
    runtime_config.phoenix_label_rebalance_interval_queries =
      static_cast<std::size_t>(phoenix_label_rebalance_interval_queries);
  }

  vecflow::scoped_runtime_config runtime_config_scope(runtime_config);

	std::vector<float> h_data;
	std::vector<float> h_queries;
	std::vector<std::vector<int>> label_data_vecs;
	std::vector<std::vector<int>> data_label_vecs;
	std::vector<std::vector<int>> query_label_vecs;
	uint32_t N, Nq, dim;
	read_labeled_data<float, int64_t>(full_data_fname, full_data_label_fname, full_query_fname, full_query_label_fname,
                                    &h_data, &h_queries,
                                    &label_data_vecs, &data_label_vecs, &query_label_vecs,
                                    &N, &Nq, &dim);

  if (!allowed_labels_file.empty()) {
    auto allowed_labels = read_allowed_labels_file(allowed_labels_file);
    filter_labels_in_place(data_label_vecs, allowed_labels);
    filter_labels_in_place(query_label_vecs, allowed_labels);
    rebuild_label_data_mapping(data_label_vecs, &label_data_vecs);
    full_ground_truth_fname = append_suffix_to_filename(
      full_ground_truth_fname,
      "_labels_" + sanitize_filename(std::filesystem::path(allowed_labels_file).stem().string()));
  }

  const auto original_query_count = static_cast<int64_t>(Nq);
  if (query_offset < 0 || query_offset > original_query_count) {
    fprintf(stderr,
            "Error: query_offset=%lld is out of range for %lld total queries.\n",
            static_cast<long long>(query_offset),
            static_cast<long long>(original_query_count));
    return 1;
  }
  const auto effective_query_count =
    (query_count < 0) ? (original_query_count - query_offset) : query_count;
  if (effective_query_count < 0 || query_offset + effective_query_count > original_query_count) {
    fprintf(stderr,
            "Error: query range [%lld, %lld) is out of range for %lld total queries.\n",
            static_cast<long long>(query_offset),
            static_cast<long long>(query_offset + effective_query_count),
            static_cast<long long>(original_query_count));
    return 1;
  }
  if (!query_id_list_file.empty()) {
    auto query_ids = read_query_id_list(query_id_list_file);
    select_query_subset_by_ids(query_ids, dim, h_queries, query_label_vecs, &Nq);
    full_ground_truth_fname = append_suffix_to_filename(
      full_ground_truth_fname,
      "_qid_" + sanitize_filename(std::filesystem::path(query_id_list_file).stem().string()));
    query_offset = 0;
  } else if (query_offset != 0 || effective_query_count != original_query_count) {
    std::vector<float> sliced_queries(static_cast<size_t>(effective_query_count) * dim);
    std::copy(h_queries.begin() + query_offset * dim,
              h_queries.begin() + (query_offset + effective_query_count) * dim,
              sliced_queries.begin());
    h_queries.swap(sliced_queries);

    std::vector<std::vector<int>> sliced_query_labels;
    sliced_query_labels.reserve(effective_query_count);
    for (int64_t i = 0; i < effective_query_count; ++i) {
      sliced_query_labels.push_back(std::move(query_label_vecs[query_offset + i]));
    }
    query_label_vecs.swap(sliced_query_labels);
    Nq = static_cast<uint32_t>(effective_query_count);
    full_ground_truth_fname = append_suffix_to_filename(
      full_ground_truth_fname,
      "_q" + std::to_string(query_offset) + "_" + std::to_string(effective_query_count));
  }

	printf("\n=== Dataset Information ===\n");
	printf("Base dataset size: N=%u, dim=%u\n", N, dim);
	printf("Query dataset size: N=%u, dim=%u\n", Nq, dim);
  printf("Query range: [%lld, %lld)\n",
         static_cast<long long>(query_offset),
         static_cast<long long>(query_offset + static_cast<int64_t>(Nq)));
  printf("Skip recall: %s\n", skip_recall ? "true" : "false");

	shared_resources::configured_raft_resources res;

	// Prepare device data
	auto stream = raft::resource::get_cuda_stream(res);
	auto d_data = raft::make_device_matrix<float, int64_t>(res, N, dim);
	raft::copy(d_data.data_handle(), h_data.data(), N * dim, stream);

	auto d_queries = raft::make_device_matrix<float, int64_t>(res, Nq, dim);
	raft::copy(d_queries.data_handle(), h_queries.data(), Nq * dim, stream);

	for (auto& labels : query_label_vecs) {
		std::sort(labels.begin(), labels.end());
		labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
	}

	// Prepare query labels for both the legacy single-label path and multi-label CSR path.
	std::vector<uint32_t> h_query_labels(Nq);
  std::vector<int64_t> h_query_label_offsets(static_cast<std::size_t>(Nq) + 1, 0);
  std::vector<uint32_t> h_query_label_indices;
  h_query_label_indices.reserve(static_cast<std::size_t>(Nq) * 2);
  int64_t filtered_out_queries = 0;
	for (int64_t i = 0; i < Nq; ++i) {
		h_query_labels[i] =
      query_label_vecs[i].empty() ? UINT32_MAX : static_cast<uint32_t>(query_label_vecs[i][0]);
    filtered_out_queries += (h_query_labels[i] == UINT32_MAX) ? 1 : 0;
    h_query_label_offsets[static_cast<std::size_t>(i)] =
      static_cast<int64_t>(h_query_label_indices.size());
    for (auto label : query_label_vecs[i]) {
      if (label < 0) { continue; }
      h_query_label_indices.push_back(static_cast<uint32_t>(label));
    }
    h_query_label_offsets[static_cast<std::size_t>(i) + 1] =
      static_cast<int64_t>(h_query_label_indices.size());
	}
  if (filtered_out_queries > 0) {
    printf("Queries with no remaining labels after filtering: %lld. "
           "VecFlow will return empty results for them.\n",
           static_cast<long long>(filtered_out_queries));
  }
	auto d_query_labels_main = raft::make_device_vector<uint32_t, int64_t>(res, Nq);
	raft::copy(d_query_labels_main.data_handle(), h_query_labels.data(), Nq, stream);

  std::optional<vecflow::multi_label_query_desc> d_multi_query_labels;
  if (use_multi_label_search) {
    d_multi_query_labels.emplace(vecflow::multi_label_query_desc{
      raft::make_device_vector<int64_t, int64_t>(res, static_cast<int64_t>(h_query_label_offsets.size())),
      raft::make_device_vector<uint32_t, int64_t>(res, static_cast<int64_t>(h_query_label_indices.size())),
      h_query_label_offsets,
      h_query_label_indices,
      multi_label_combine_mode,
      multi_label_and_mode});
    raft::copy(d_multi_query_labels->label_offsets.data_handle(),
               h_query_label_offsets.data(),
               static_cast<int64_t>(h_query_label_offsets.size()),
               stream);
    if (!h_query_label_indices.empty()) {
      raft::copy(d_multi_query_labels->label_indices.data_handle(),
                 h_query_label_indices.data(),
                 static_cast<int64_t>(h_query_label_indices.size()),
                 stream);
    }
  }

  std::optional<raft::device_matrix<uint32_t, int64_t>> gt_neighbors = std::nullopt;
  std::vector<uint32_t> h_gt_neighbors;
  if (!skip_recall) {
    gt_neighbors.emplace(raft::make_device_matrix<uint32_t, int64_t>(res, Nq, topk));
	  generate_ground_truth(res,
									raft::make_const_mdspan(d_data.view()),
									raft::make_const_mdspan(d_queries.view()),
									label_data_vecs,
									query_label_vecs,
									gt_neighbors->view(),
									full_ground_truth_fname,
                          ground_truth_label_mode);
    h_gt_neighbors.resize(static_cast<size_t>(Nq) * topk);
    raft::copy(h_gt_neighbors.data(),
               gt_neighbors->data_handle(),
               static_cast<int64_t>(Nq) * topk,
               stream);
    raft::resource::sync_stream(res);
  } else {
    printf("Skipping ground truth generation and recall computation as configured.\n");
  }

	// Initialize the JSON array for results
	json results_json = json::array();

  auto run_vecflow_benchmark = [&](const std::string& algorithm_name,
                                   vecflow::graph_builder_type builder,
                                   const std::string& graph_fname) {
    printf("\n=== %s Index Building and Search ===\n", algorithm_name.c_str());
    auto build_start = std::chrono::high_resolution_clock::now();
    auto idx = vecflow::build(res,
                              raft::make_const_mdspan(d_data.view()),
                              data_label_vecs,
                              graph_degree,
                              specificity_threshold,
                              graph_fname,
                              full_ivf_bfs_fname,
                              force_rebuild,
                              builder,
                              tagore_iterations);
    raft::resource::sync_stream(res);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_seconds = std::chrono::duration<double>(build_end - build_start).count();
    printf("Build time: %.3f s\n", build_seconds);

    auto vecflow_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, Nq, topk);
    auto vecflow_distances = raft::make_device_matrix<float, int64_t>(res, Nq, topk);

    auto run_search_once = [&](int current_itopk) {
      if (use_multi_label_search) {
        if (!d_multi_query_labels.has_value()) {
          throw std::runtime_error("query_label_mode requires multi-label query descriptors");
        }
        vecflow::search(res,
                        idx,
                        raft::make_const_mdspan(d_queries.view()),
                        *d_multi_query_labels,
                        current_itopk,
                        vecflow_neighbors.view(),
                        vecflow_distances.view());
      } else {
        vecflow::search(res,
                        idx,
                        raft::make_const_mdspan(d_queries.view()),
                        d_query_labels_main.view(),
                        current_itopk,
                        vecflow_neighbors.view(),
                        vecflow_distances.view());
      }
      raft::resource::sync_stream(res);
    };

    printf("\n=== %s Search Benchmarking ===\n", algorithm_name.c_str());
    for (int current_itopk : itopk_sizes) {
      printf("-- Running %s Search (itopk=%d) --\n", algorithm_name.c_str(), current_itopk);
      for (int i = 0; i < warmup_runs; i++) {
        run_search_once(current_itopk);
      }

      auto cache_counters_before = capture_vecflow_cache_counters(idx);
      std::vector<double> run_latencies_ms;
      std::vector<double> run_gpu_latencies_ms;
      std::vector<double> run_qps_samples;
      run_latencies_ms.reserve(num_runs);
      run_gpu_latencies_ms.reserve(num_runs);
      run_qps_samples.reserve(num_runs);
      cudaEvent_t gpu_start = nullptr;
      cudaEvent_t gpu_end = nullptr;
      RAFT_CUDA_TRY(cudaEventCreate(&gpu_start));
      RAFT_CUDA_TRY(cudaEventCreate(&gpu_end));
      dynamic_stability_result dynamic_stability;
      bool dynamic_stop_triggered = false;
      std::string dynamic_stop_reason = "reached max_num_runs";
#ifdef VECFLOW_BENCH_PROGRESS_LOG
      int64_t progress_total_queries = 0;
      int64_t progress_next_milestone = 0;
      {
        auto ts_now = std::chrono::system_clock::now();
        auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
          ts_now.time_since_epoch()).count();
        fprintf(stderr, "PROGRESS q=0 ts_us=%" PRId64 "\n", ts_us);
      }
#endif
      for (int i = 0; i < num_runs; i++) {
        auto run_start = std::chrono::high_resolution_clock::now();
        RAFT_CUDA_TRY(cudaEventRecord(gpu_start, stream));
        run_search_once(current_itopk);
        RAFT_CUDA_TRY(cudaEventRecord(gpu_end, stream));
        RAFT_CUDA_TRY(cudaEventSynchronize(gpu_end));
        auto run_end = std::chrono::high_resolution_clock::now();
#ifdef VECFLOW_BENCH_PROGRESS_LOG
        progress_total_queries += static_cast<int64_t>(Nq);
        if (progress_total_queries >= progress_next_milestone) {
          auto ts_now = std::chrono::system_clock::now();
          auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
            ts_now.time_since_epoch()).count();
          fprintf(stderr, "PROGRESS q=%" PRId64 " ts_us=%" PRId64 "\n",
                  progress_total_queries, ts_us);
          progress_next_milestone = ((progress_total_queries / 10000) + 1) * 10000;
        }
#endif
        auto run_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
        float gpu_ms = 0.0f;
        RAFT_CUDA_TRY(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_end));
        run_latencies_ms.push_back(run_ms);
        run_gpu_latencies_ms.push_back(static_cast<double>(gpu_ms));
        auto run_seconds = run_ms / 1000.0;
        auto run_qps = run_seconds > 0.0 ? static_cast<double>(Nq) / run_seconds : 0.0;
        run_qps_samples.push_back(run_qps);
        printf("  - Run %d/%d: qps=%.2f latency_ms=%.3f gpu_ms=%.3f\n",
               i + 1,
               num_runs,
               run_qps,
               run_ms,
               static_cast<double>(gpu_ms));

        dynamic_stability = evaluate_dynamic_stability(run_qps_samples, run_latencies_ms, dynamic_policy);
        if (dynamic_stability.triggered && static_cast<int>(run_latencies_ms.size()) < num_runs) {
          dynamic_stop_triggered = true;
          dynamic_stop_reason = "stability criteria satisfied";
          printf("  - Early stop triggered after %zu runs. Stable subset runs: [",
                 run_latencies_ms.size());
          for (std::size_t subset_idx = 0;
               subset_idx < dynamic_stability.subset_run_numbers.size();
               ++subset_idx) {
            printf("%d%s",
                   dynamic_stability.subset_run_numbers[subset_idx],
                   subset_idx + 1 == dynamic_stability.subset_run_numbers.size() ? "" : " ");
          }
          printf("], qps span=%.2f%%, latency span=%.2f%%\n",
                 dynamic_stability.qps_rel_span * 100.0,
                 dynamic_stability.latency_rel_span * 100.0);
          break;
        }
      }
      RAFT_CUDA_TRY(cudaEventDestroy(gpu_start));
      RAFT_CUDA_TRY(cudaEventDestroy(gpu_end));

      auto cache_counters_after = capture_vecflow_cache_counters(idx);
      auto cache_counter_delta =
        subtract_vecflow_cache_counters(cache_counters_after, cache_counters_before);
      auto latency_stats = summarize_latencies_ms(run_latencies_ms);
      auto gpu_latency_stats = summarize_latencies_ms(run_gpu_latencies_ms);
      std::vector<double> cpu_overhead_samples_ms;
      cpu_overhead_samples_ms.reserve(run_latencies_ms.size());
      for (std::size_t sample = 0; sample < run_latencies_ms.size(); ++sample) {
        cpu_overhead_samples_ms.push_back(
          std::max(0.0, run_latencies_ms[sample] - run_gpu_latencies_ms[sample]));
      }
      auto cpu_overhead_stats = summarize_latencies_ms(cpu_overhead_samples_ms);
      auto total_time_seconds = latency_stats.total_ms / 1000.0;
      double qps = total_time_seconds > 0.0
                     ? static_cast<double>(num_runs) * static_cast<double>(Nq) / total_time_seconds
                     : 0.0;
      double recall = (!skip_recall && Nq > 0)
                        ? compute_recall(res, vecflow_neighbors.view(), gt_neighbors->view())
                        : 0.0;
      auto storage_stats = vecflow::storage_stats(idx);
      auto actual_num_runs = static_cast<int>(run_latencies_ms.size());
      if (skip_recall) {
        printf("  - QPS: %.2f, Recall@%d: skipped\n", qps, topk);
      } else {
        printf("  - QPS: %.2f, Recall@%d: %.4f\n", qps, topk, recall);
      }
      printf("  - Executed runs: %d (min=%d, max=%d)\n",
             actual_num_runs,
             min_num_runs,
             max_num_runs);
      printf("  - Latency(ms): total avg=%.3f p50=%.3f p95=%.3f max=%.3f\n",
             latency_stats.avg_ms,
             latency_stats.p50_ms,
             latency_stats.p95_ms,
             latency_stats.max_ms);
      printf("  - GPU(ms): avg=%.3f p50=%.3f p95=%.3f max=%.3f\n",
             gpu_latency_stats.avg_ms,
             gpu_latency_stats.p50_ms,
             gpu_latency_stats.p95_ms,
             gpu_latency_stats.max_ms);
      printf("  - CPU-overhead(ms): avg=%.3f p50=%.3f p95=%.3f max=%.3f\n",
             cpu_overhead_stats.avg_ms,
             cpu_overhead_stats.p50_ms,
             cpu_overhead_stats.p95_ms,
             cpu_overhead_stats.max_ms);
      printf("  - Cache delta: graph[hbm=%llu dram=%llu ssd=%llu] dataset[hbm=%llu dram=%llu ssd=%llu] bfs[hbm=%llu dram=%llu ssd=%llu]\n",
             static_cast<unsigned long long>(cache_counter_delta.phoenix_graph_hbm_hits),
             static_cast<unsigned long long>(cache_counter_delta.phoenix_graph_dram_hits),
             static_cast<unsigned long long>(cache_counter_delta.phoenix_graph_ssd_loads),
             static_cast<unsigned long long>(cache_counter_delta.phoenix_dataset_hbm_hits),
             static_cast<unsigned long long>(cache_counter_delta.phoenix_dataset_dram_hits),
             static_cast<unsigned long long>(cache_counter_delta.phoenix_dataset_ssd_loads),
             static_cast<unsigned long long>(cache_counter_delta.bfs_hbm_hits),
             static_cast<unsigned long long>(cache_counter_delta.bfs_dram_hits),
             static_cast<unsigned long long>(cache_counter_delta.bfs_ssd_loads));
      if (!skip_recall && Nq > 0) {
        auto h_neighbors_dbg = raft::make_host_matrix<uint32_t, int64_t>(1, topk);
        auto h_gt_dbg = raft::make_host_matrix<uint32_t, int64_t>(1, topk);
        raft::copy(h_neighbors_dbg.data_handle(),
                   vecflow_neighbors.data_handle(),
                   topk,
                   raft::resource::get_cuda_stream(res));
        raft::copy(h_gt_dbg.data_handle(),
                   gt_neighbors->data_handle(),
                   topk,
                   raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);
        printf("  - First query neighbors: ");
        for (int j = 0; j < topk; ++j) { printf("%u ", h_neighbors_dbg.view()(0, j)); }
        printf("\n");
        printf("  - First query ground truth: ");
        for (int j = 0; j < topk; ++j) { printf("%u ", h_gt_dbg.view()(0, j)); }
        printf("\n");
      }
      results_json.push_back({{"algorithm", algorithm_name},
                              {"itopk", current_itopk},
                              {"qps", qps},
                              {"recall", skip_recall ? json(nullptr) : json(recall)},
                              {"skip_recall", skip_recall},
                              {"search_seconds", total_time_seconds},
                              {"latency_ms", latency_summary_to_json(latency_stats)},
                              {"gpu_latency_ms", latency_summary_to_json(gpu_latency_stats)},
                              {"cpu_overhead_ms", latency_summary_to_json(cpu_overhead_stats)},
                              {"cache_counters", vecflow_cache_counters_to_json(cache_counter_delta)},
                              {"storage_stats", storage_stats_info_to_json(storage_stats)},
                              {"build_seconds", build_seconds},
                              {"num_runs", actual_num_runs},
                              {"num_runs_requested", max_num_runs},
                              {"min_num_runs", min_num_runs},
                              {"max_num_runs", max_num_runs},
                              {"warmup_runs", warmup_runs},
                              {"stability_window", stability_window},
                              {"stable_subset_size", stable_subset_size},
                              {"qps_stability_rel_tol", qps_stability_rel_tol},
                              {"latency_stability_rel_tol", latency_stability_rel_tol},
                              {"trend_guard_rel_tol", trend_guard_rel_tol},
                              {"dynamic_stop_triggered", dynamic_stop_triggered},
                              {"dynamic_stop_reason", dynamic_stop_reason},
                              {"dynamic_stop_subset_runs", dynamic_stability.subset_run_numbers},
                              {"dynamic_stop_subset_mean_qps", dynamic_stability.subset_mean_qps},
                              {"dynamic_stop_subset_mean_latency_ms",
                               dynamic_stability.subset_mean_latency_ms},
                              {"dynamic_stop_qps_rel_span", dynamic_stability.qps_rel_span},
                              {"dynamic_stop_latency_rel_span",
                               dynamic_stability.latency_rel_span},
                              {"run_qps_samples", run_qps_samples},
                              {"run_latency_ms_samples", run_latencies_ms},
                              {"run_gpu_latency_ms_samples", run_gpu_latencies_ms},
                              {"num_queries", static_cast<int64_t>(Nq)},
                              {"query_offset", query_offset},
                              {"query_label_mode", query_label_mode},
                              {"ground_truth_label_mode",
                               ground_truth_label_mode == query_label_match_mode::ALL ? "all"
                                                                                      : "any"}});
    }
  };

  auto run_vecflow_mg_benchmark = [&](const std::string& algorithm_name,
                                      vecflow::graph_builder_type builder,
                                      const std::string& graph_fname) {
    printf("\n=== %s Multi-GPU Index Building and Search ===\n", algorithm_name.c_str());
    auto build_start = std::chrono::high_resolution_clock::now();
    auto contexts = build_vecflow_mg_contexts(device_ids,
                                              h_data,
                                              N,
                                              dim,
                                              data_label_vecs,
                                              graph_degree,
                                              specificity_threshold,
                                              graph_fname,
                                              full_ivf_bfs_fname,
                                              force_rebuild,
                                              builder,
                                              tagore_iterations);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_seconds = std::chrono::duration<double>(build_end - build_start).count();
    printf("Build time: %.3f s\n", build_seconds);

    auto chunks = split_query_chunks(Nq, static_cast<int>(contexts.size()));
    std::vector<uint32_t> h_neighbors(static_cast<size_t>(Nq) * topk, UINT32_MAX);
    std::vector<float> h_distances(static_cast<size_t>(Nq) * topk, std::numeric_limits<float>::infinity());

    auto run_once = [&](int current_itopk) {
      std::vector<std::thread> workers;
      workers.reserve(contexts.size());
      for (size_t worker = 0; worker < contexts.size(); ++worker) {
        workers.emplace_back([&, worker]() {
          search_vecflow_mg_chunk(contexts[worker],
                                  static_cast<int>(worker),
                                  static_cast<int>(contexts.size()),
                                  h_queries,
                                  h_query_labels,
                                  dim,
                                  chunks[worker].offset,
                                  chunks[worker].size,
                                  current_itopk,
                                  topk,
                                  h_neighbors,
                                  h_distances);
        });
      }
      for (auto& worker : workers) { worker.join(); }
    };

    printf("\n=== %s Multi-GPU Search Benchmarking ===\n", algorithm_name.c_str());
    for (int current_itopk : itopk_sizes) {
      printf("-- Running %s Multi-GPU Search (itopk=%d) --\n", algorithm_name.c_str(), current_itopk);
      for (int i = 0; i < warmup_runs; ++i) { run_once(current_itopk); }

      auto start_time = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < num_runs; ++i) { run_once(current_itopk); }
      auto end_time = std::chrono::high_resolution_clock::now();

      auto total_time = std::chrono::duration<double>(end_time - start_time).count();
      double qps = num_runs * static_cast<double>(Nq) / total_time;
      double recall = (!skip_recall && Nq > 0) ? compute_recall_host(h_neighbors, h_gt_neighbors, Nq, topk)
                                               : 0.0;
      if (skip_recall) {
        printf("  - QPS: %.2f, Recall@%d: skipped\n", qps, topk);
      } else {
        printf("  - QPS: %.2f, Recall@%d: %.4f\n", qps, topk, recall);
      }
      if (!skip_recall && Nq > 0) {
        printf("  - First query neighbors: ");
        for (int j = 0; j < topk; ++j) { printf("%u ", h_neighbors[j]); }
        printf("\n");
        printf("  - First query ground truth: ");
        for (int j = 0; j < topk; ++j) { printf("%u ", h_gt_neighbors[j]); }
        printf("\n");
      }
      results_json.push_back({{"algorithm", algorithm_name},
                              {"itopk", current_itopk},
                              {"qps", qps},
                              {"recall", recall},
                              {"build_seconds", build_seconds},
                              {"num_devices", static_cast<int>(device_ids.size())}});
    }
  };

	if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "vecflow") != algorithms_to_run.end()) {
    run_vecflow_benchmark("vecflow", vecflow::graph_builder_type::CAGRA, full_ivf_graph_fname);
	}

  if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "vecflow_tagore") != algorithms_to_run.end()) {
    run_vecflow_benchmark(
      "vecflow_tagore",
      vecflow::graph_builder_type::TAGORE_CAGRA_COMPAT,
      full_ivf_graph_tagore_fname);
  }

  if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "vecflow_mg") != algorithms_to_run.end()) {
    run_vecflow_mg_benchmark("vecflow_mg",
                             vecflow::graph_builder_type::CAGRA,
                             full_ivf_graph_fname);
  }

  if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "vecflow_tagore_mg") != algorithms_to_run.end()) {
    run_vecflow_mg_benchmark("vecflow_tagore_mg",
                             vecflow::graph_builder_type::TAGORE_CAGRA_COMPAT,
                             full_ivf_graph_tagore_fname);
  }

	// CAGRA Post-Processing
  cagra::index<float, uint32_t> cagra_index(res);
	if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "cagra_post_processing") != algorithms_to_run.end()) {
		// Build CAGRA index if not already built
		printf("\n=== Building CAGRA Index (for Post-Processing) ===\n");
		build_cagra_index(res, cagra_index, raft::make_const_mdspan(d_data.view()),
						          full_cagra_index_fname, graph_degree);

		printf("\n=== CAGRA Search with Post-Processing Benchmarking ===\n");
		auto filtered_neighbors_pp = raft::make_device_matrix<uint32_t, int64_t>(res, Nq, topk);
		for (int current_itopk : itopk_sizes) {
			printf("-- Running CAGRA Search + PP (itopk=%d) --\n", current_itopk);
			double qps = cagra_search_with_post_processing(res, cagra_index,
										raft::make_const_mdspan(d_queries.view()),
										d_query_labels_main.view(), label_data_vecs,
										label_data_vecs.size(), current_itopk, topk,
										num_runs, warmup_runs, filtered_neighbors_pp.view());
			double recall = (!skip_recall && Nq > 0)
			                  ? compute_recall(res, filtered_neighbors_pp.view(), gt_neighbors->view())
			                  : 0.0;
			if (skip_recall) {
				printf("  - QPS: %.2f, Recall@%d: skipped\n", qps, topk);
			} else {
				printf("  - QPS: %.2f, Recall@%d: %.4f\n", qps, topk, recall);
			}
      if (!skip_recall && Nq > 0) {
        auto h_neighbors_dbg = raft::make_host_matrix<uint32_t, int64_t>(1, topk);
        auto h_gt_dbg = raft::make_host_matrix<uint32_t, int64_t>(1, topk);
        raft::copy(h_neighbors_dbg.data_handle(),
                   filtered_neighbors_pp.data_handle(),
                   topk,
                   raft::resource::get_cuda_stream(res));
        raft::copy(h_gt_dbg.data_handle(),
                   gt_neighbors->data_handle(),
                   topk,
                   raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);
        printf("  - First query neighbors: ");
        for (int j = 0; j < topk; ++j) { printf("%u ", h_neighbors_dbg.view()(0, j)); }
        printf("\n");
        printf("  - First query ground truth: ");
        for (int j = 0; j < topk; ++j) { printf("%u ", h_gt_dbg.view()(0, j)); }
        printf("\n");
      }
			results_json.push_back({{"algorithm", "cagra_post_processing"},
                              {"itopk", current_itopk},
                              {"qps", qps},
                              {"recall", skip_recall ? json(nullptr) : json(recall)},
                              {"skip_recall", skip_recall},
                              {"num_queries", static_cast<int64_t>(Nq)},
                              {"query_offset", query_offset}});
		}
	}

	// CAGRA Inline Filtering
	if (std::find(algorithms_to_run.begin(), algorithms_to_run.end(), "cagra_inline_filtering") != algorithms_to_run.end()) {
    printf("\n=== Building CAGRA Index (for Inline Filtering) ===\n");
    build_cagra_index(res, cagra_index, raft::make_const_mdspan(d_data.view()),
              full_cagra_index_fname, graph_degree);

    printf("\n=== CAGRA Search with Inline Filtering Benchmarking ===\n");
    auto filtered_neighbors_inline = raft::make_device_matrix<uint32_t, int64_t>(res, Nq, topk);
    for (int current_itopk : itopk_sizes) {
      printf("-- Running CAGRA Search + Inline Filter (itopk=%d) --\n", current_itopk);
      double qps = cagra_search_inline_filtering(res, cagra_index,
                            raft::make_const_mdspan(d_queries.view()),
                            d_query_labels_main.view(), label_data_vecs,
                            current_itopk, topk, num_runs, warmup_runs,
                            filtered_neighbors_inline.view());
      double recall = (!skip_recall && Nq > 0)
                        ? compute_recall(res, filtered_neighbors_inline.view(), gt_neighbors->view())
                        : 0.0;
      if (skip_recall) {
        printf("  - QPS: %.2f, Recall@%d: skipped\n", qps, topk);
      } else {
        printf("  - QPS: %.2f, Recall@%d: %.4f\n", qps, topk, recall);
      }
      if (!skip_recall && Nq > 0) {
        auto h_neighbors_dbg = raft::make_host_matrix<uint32_t, int64_t>(1, topk);
        auto h_gt_dbg = raft::make_host_matrix<uint32_t, int64_t>(1, topk);
        raft::copy(h_neighbors_dbg.data_handle(),
                   filtered_neighbors_inline.data_handle(),
                   topk,
                   raft::resource::get_cuda_stream(res));
        raft::copy(h_gt_dbg.data_handle(),
                   gt_neighbors->data_handle(),
                   topk,
                   raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);
        printf("  - First query neighbors: ");
        for (int j = 0; j < topk; ++j) { printf("%u ", h_neighbors_dbg.view()(0, j)); }
        printf("\n");
        printf("  - First query ground truth: ");
        for (int j = 0; j < topk; ++j) { printf("%u ", h_gt_dbg.view()(0, j)); }
        printf("\n");
      }
      results_json.push_back({{"algorithm", "cagra_inline_filtering"},
                              {"itopk", current_itopk},
                              {"qps", qps},
                              {"recall", skip_recall ? json(nullptr) : json(recall)},
                              {"skip_recall", skip_recall},
                              {"num_queries", static_cast<int64_t>(Nq)},
                              {"query_offset", query_offset}});
    }
	}

	// --- Write Results to JSON ---
	printf("\nWriting results to %s\n", output_json_file.c_str());
	std::ofstream output_file(output_json_file);
	if (output_file.is_open()) {
		output_file << results_json.dump(2); // Indent with 2 spaces
		output_file.close();
		printf("Results successfully written.\n");
	} else {
		fprintf(stderr, "Error: Unable to open output file: %s\n", output_json_file.c_str());
	}

	return 0;
}
