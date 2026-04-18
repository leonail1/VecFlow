#include <cuvs/neighbors/vecflow.hpp>
#include <cuvs/neighbors/shared_resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "../common.cuh"

using namespace cuvs::neighbors;
using json = nlohmann::json;

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
    snapshot.phoenix_dataset_dram_evictions = index.phoenix_label_dataset_cache->dram_evictions;
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

struct telemetry_sample {
  double elapsed_seconds = 0.0;
  std::int64_t queries_completed = 0;
  vecflow_cache_counters_snapshot cache_counters;
};

json telemetry_sample_to_json(const telemetry_sample& sample)
{
  return json{{"elapsed_seconds", sample.elapsed_seconds},
              {"queries_completed", sample.queries_completed},
              {"cache_counters", vecflow_cache_counters_to_json(sample.cache_counters)}};
}

std::string append_suffix_to_filename(const std::string& filename, const std::string& suffix)
{
  auto path = std::filesystem::path(filename);
  auto stem = path.stem().string();
  auto ext = path.extension().string();
  auto parent = path.parent_path();
  return (parent / (stem + suffix + ext)).string();
}

vecflow::graph_builder_type resolve_graph_builder(const json& config)
{
  if (config.contains("graph_builder")) {
    auto builder = config["graph_builder"].get<std::string>();
    if (builder == "tagore" || builder == "tagore_cagra" || builder == "tagore_cagra_compat") {
      return vecflow::graph_builder_type::TAGORE_CAGRA_COMPAT;
    }
  }
  if (config.contains("algorithms_to_run")) {
    auto algorithms = config["algorithms_to_run"].get<std::vector<std::string>>();
    for (auto const& algorithm : algorithms) {
      if (algorithm == "vecflow_tagore") { return vecflow::graph_builder_type::TAGORE_CAGRA_COMPAT; }
    }
  }
  return vecflow::graph_builder_type::CAGRA;
}

int resolve_itopk(const json& config)
{
  if (!config.contains("itopk_size")) {
    throw std::runtime_error("Config must contain itopk_size");
  }
  if (config["itopk_size"].is_array()) {
    auto values = config["itopk_size"].get<std::vector<int>>();
    if (values.empty()) {
      throw std::runtime_error("Config itopk_size array is empty");
    }
    return values.front();
  }
  return config["itopk_size"].get<int>();
}

int main(int argc, char** argv)
{
  std::string config_file;
  if (argc < 3 || std::string(argv[1]) != "--config") {
    fprintf(stderr, "Usage: %s --config <config.json>\n", argv[0]);
    return 1;
  }
  config_file = argv[2];

  std::ifstream file(config_file);
  if (!file.is_open()) {
    fprintf(stderr, "Unable to open config file: %s\n", config_file.c_str());
    return 1;
  }

  std::string data_dir;
  std::string data_fname;
  std::string query_fname;
  std::string data_label_fname;
  std::string query_label_fname;
  std::string ivf_graph_fname;
  std::string ivf_graph_tagore_fname;
  std::string ivf_bfs_fname;
  std::string output_json_file;
  std::string query_label_mode = "single";
  std::int64_t query_offset = 0;
  std::int64_t query_count = -1;
  int graph_degree = 16;
  int specificity_threshold = 1000;
  int topk = 10;
  int itopk_size = 64;
  int num_runs = 10;
  int warmup_runs = 5;
  int tagore_iterations = 10;
  bool force_rebuild = false;
  bool use_phoenix_graph_load = false;
  bool use_phoenix_label_load = false;
  bool enable_bfs_tiered_cache = false;
  bool cascade_eviction = false;
  std::uint64_t phoenix_label_cache_bytes = 1ULL << 30;
  std::uint64_t phoenix_label_dram_cache_bytes = 0;
  std::uint64_t phoenix_label_prefetch_max_bytes = 0;
  std::uint64_t phoenix_label_dataset_cache_bytes = 1ULL << 30;
  std::uint64_t phoenix_label_dataset_dram_cache_bytes = 0;
  std::uint64_t phoenix_label_dataset_prefetch_max_bytes = 0;
  std::uint64_t phoenix_label_rebalance_interval_queries = 64;
  std::uint64_t bfs_hbm_cache_bytes = 0;
  std::uint64_t bfs_dram_cache_bytes = 0;
  std::uint64_t bfs_prefetch_max_bytes = 0;
  std::uint64_t bfs_rebalance_interval_queries = 64;
  std::int64_t telemetry_query_chunk_size = 16384;

  try {
    json config;
    file >> config;
    data_dir = config["data_dir"];
    data_fname = config["data_fname"];
    query_fname = config["query_fname"];
    data_label_fname = config["data_label_fname"];
    query_label_fname = config["query_label_fname"];
    ivf_graph_fname = config["ivf_graph_fname"];
    ivf_graph_tagore_fname = config.value("ivf_graph_tagore_fname", ivf_graph_fname);
    ivf_bfs_fname = config["ivf_bfs_fname"];
    output_json_file = config["output_json_file"];
    graph_degree = config["graph_degree"];
    specificity_threshold = config["spec_threshold"];
    force_rebuild = config.value("force_rebuild", false);
    itopk_size = resolve_itopk(config);
    topk = config["topk"];
    num_runs = config["num_runs"];
    warmup_runs = config["warmup_runs"];
    tagore_iterations = config.value("tagore_iterations", 10);
    query_label_mode = config.value("query_label_mode", std::string("single"));
    query_offset = config.value("query_offset", static_cast<std::int64_t>(0));
    query_count = config.value("query_count", static_cast<std::int64_t>(-1));
    use_phoenix_graph_load = config.value("use_phoenix_graph_load", false);
    use_phoenix_label_load = config.value("use_phoenix_label_load", false);
    enable_bfs_tiered_cache = config.value("enable_bfs_tiered_cache", false);
    cascade_eviction = config.value("cascade_eviction", false);
    phoenix_label_cache_bytes = config.value("phoenix_label_cache_bytes", phoenix_label_cache_bytes);
    phoenix_label_dram_cache_bytes = config.value("phoenix_label_dram_cache_bytes", phoenix_label_dram_cache_bytes);
    phoenix_label_prefetch_max_bytes = config.value("phoenix_label_prefetch_max_bytes", phoenix_label_prefetch_max_bytes);
    phoenix_label_dataset_cache_bytes = config.value("phoenix_label_dataset_cache_bytes", phoenix_label_dataset_cache_bytes);
    phoenix_label_dataset_dram_cache_bytes = config.value("phoenix_label_dataset_dram_cache_bytes", phoenix_label_dataset_dram_cache_bytes);
    phoenix_label_dataset_prefetch_max_bytes = config.value("phoenix_label_dataset_prefetch_max_bytes", phoenix_label_dataset_prefetch_max_bytes);
    phoenix_label_rebalance_interval_queries = config.value("phoenix_label_rebalance_interval_queries", phoenix_label_rebalance_interval_queries);
    bfs_hbm_cache_bytes = config.value("bfs_hbm_cache_bytes", bfs_hbm_cache_bytes);
    bfs_dram_cache_bytes = config.value("bfs_dram_cache_bytes", bfs_dram_cache_bytes);
    bfs_prefetch_max_bytes = config.value("bfs_prefetch_max_bytes", bfs_prefetch_max_bytes);
    bfs_rebalance_interval_queries = config.value("bfs_rebalance_interval_queries", bfs_rebalance_interval_queries);
    telemetry_query_chunk_size = config.value("telemetry_query_chunk_size", telemetry_query_chunk_size);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error parsing JSON config file: %s\n", e.what());
    return 1;
  }

  if (query_label_mode != "single") {
    fprintf(stderr,
            "VECFLOW_QPS_TIMESERIES currently supports only query_label_mode=single; got %s\n",
            query_label_mode.c_str());
    return 1;
  }

  auto builder = resolve_graph_builder(([&]() { std::ifstream second(config_file); json parsed; second >> parsed; return parsed; })());
  std::string full_data_fname = data_dir + data_fname;
  std::string full_query_fname = data_dir + query_fname;
  std::string full_data_label_fname = data_dir + data_label_fname;
  std::string full_query_label_fname = data_dir + query_label_fname;
  std::string full_ivf_graph_fname =
    data_dir + (builder == vecflow::graph_builder_type::TAGORE_CAGRA_COMPAT
                  ? ivf_graph_tagore_fname
                  : ivf_graph_fname);
  std::string full_ivf_bfs_fname = data_dir + ivf_bfs_fname;

  printf("Loading configuration from %s\n", config_file.c_str());
  printf("\n=== Time-Series Configuration ===\n");
  printf("Query label mode: %s\n", query_label_mode.c_str());
  printf("iTopK size: %d\n", itopk_size);
  printf("TopK: %d\n", topk);
  printf("Number of runs: %d\n", num_runs);
  printf("Warmup runs: %d\n", warmup_runs);
  printf("Telemetry query chunk size: %lld\n", static_cast<long long>(telemetry_query_chunk_size));
  printf("Output JSON file: %s\n", output_json_file.c_str());

  std::vector<float> h_data;
  std::vector<float> h_queries;
  std::vector<std::vector<int>> label_data_vecs;
  std::vector<std::vector<int>> data_label_vecs;
  std::vector<std::vector<int>> query_label_vecs;
  uint32_t N, Nq, dim;
  read_labeled_data<float, int64_t>(full_data_fname,
                                    full_data_label_fname,
                                    full_query_fname,
                                    full_query_label_fname,
                                    &h_data,
                                    &h_queries,
                                    &label_data_vecs,
                                    &data_label_vecs,
                                    &query_label_vecs,
                                    &N,
                                    &Nq,
                                    &dim);

  const auto original_query_count = static_cast<int64_t>(Nq);
  if (query_offset < 0 || query_offset > original_query_count) {
    fprintf(stderr, "query_offset=%lld is out of range for %lld total queries\n",
            static_cast<long long>(query_offset),
            static_cast<long long>(original_query_count));
    return 1;
  }
  const auto effective_query_count =
    (query_count < 0) ? (original_query_count - query_offset) : query_count;
  if (effective_query_count < 0 || query_offset + effective_query_count > original_query_count) {
    fprintf(stderr,
            "query range [%lld, %lld) is out of range for %lld total queries\n",
            static_cast<long long>(query_offset),
            static_cast<long long>(query_offset + effective_query_count),
            static_cast<long long>(original_query_count));
    return 1;
  }
  if (query_offset != 0 || effective_query_count != original_query_count) {
    std::vector<float> sliced_queries(static_cast<size_t>(effective_query_count) * dim);
    std::copy(h_queries.begin() + query_offset * dim,
              h_queries.begin() + (query_offset + effective_query_count) * dim,
              sliced_queries.begin());
    h_queries.swap(sliced_queries);

    std::vector<std::vector<int>> sliced_query_labels;
    sliced_query_labels.reserve(effective_query_count);
    for (int64_t i = 0; i < effective_query_count; ++i) {
      sliced_query_labels.push_back(query_label_vecs[query_offset + i]);
    }
    query_label_vecs.swap(sliced_query_labels);
    Nq = static_cast<uint32_t>(effective_query_count);
  }

  printf("\n=== Dataset Information ===\n");
  printf("Base dataset size: N=%u, dim=%u\n", N, dim);
  printf("Query dataset size: N=%u, dim=%u\n", Nq, dim);

  shared_resources::configured_raft_resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  auto d_data = raft::make_device_matrix<float, int64_t>(res, N, dim);
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, Nq, dim);
  raft::copy(d_data.data_handle(), h_data.data(), static_cast<int64_t>(N) * dim, stream);
  raft::copy(d_queries.data_handle(), h_queries.data(), static_cast<int64_t>(Nq) * dim, stream);

  std::vector<uint32_t> h_query_labels(Nq, UINT32_MAX);
  for (int64_t i = 0; i < static_cast<int64_t>(Nq); ++i) {
    if (!query_label_vecs[static_cast<std::size_t>(i)].empty()) {
      h_query_labels[static_cast<std::size_t>(i)] =
        static_cast<uint32_t>(query_label_vecs[static_cast<std::size_t>(i)][0]);
    }
  }
  auto d_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, Nq);
  raft::copy(d_query_labels.data_handle(), h_query_labels.data(), static_cast<int64_t>(Nq), stream);
  raft::resource::sync_stream(res);

  vecflow::runtime_config runtime_config;
  runtime_config.cascade_eviction = cascade_eviction;
  runtime_config.enable_bfs_tiered_cache = enable_bfs_tiered_cache;
  runtime_config.use_phoenix_label_load = use_phoenix_label_load;
  runtime_config.use_phoenix_graph_load = use_phoenix_label_load ? false : use_phoenix_graph_load;
  if (enable_bfs_tiered_cache) {
    runtime_config.bfs_hbm_cache_bytes = static_cast<std::size_t>(bfs_hbm_cache_bytes);
    runtime_config.bfs_dram_cache_bytes = static_cast<std::size_t>(bfs_dram_cache_bytes);
    runtime_config.bfs_prefetch_max_bytes = static_cast<std::size_t>(bfs_prefetch_max_bytes);
    runtime_config.bfs_rebalance_interval_queries =
      static_cast<std::size_t>(bfs_rebalance_interval_queries);
  }
  if (use_phoenix_label_load) {
    runtime_config.phoenix_label_cache_bytes = static_cast<std::size_t>(phoenix_label_cache_bytes);
    runtime_config.phoenix_label_dram_cache_bytes = static_cast<std::size_t>(phoenix_label_dram_cache_bytes);
    runtime_config.phoenix_label_prefetch_max_bytes = static_cast<std::size_t>(phoenix_label_prefetch_max_bytes);
    runtime_config.phoenix_label_dataset_cache_bytes = static_cast<std::size_t>(phoenix_label_dataset_cache_bytes);
    runtime_config.phoenix_label_dataset_dram_cache_bytes = static_cast<std::size_t>(phoenix_label_dataset_dram_cache_bytes);
    runtime_config.phoenix_label_dataset_prefetch_max_bytes = static_cast<std::size_t>(phoenix_label_dataset_prefetch_max_bytes);
    runtime_config.phoenix_label_rebalance_interval_queries =
      static_cast<std::size_t>(phoenix_label_rebalance_interval_queries);
  }
  vecflow::scoped_runtime_config runtime_config_scope(runtime_config);

  auto build_start = std::chrono::high_resolution_clock::now();
  auto idx = vecflow::build(res,
                            raft::make_const_mdspan(d_data.view()),
                            data_label_vecs,
                            graph_degree,
                            specificity_threshold,
                            full_ivf_graph_fname,
                            full_ivf_bfs_fname,
                            force_rebuild,
                            builder,
                            tagore_iterations);
  raft::resource::sync_stream(res);
  auto build_end = std::chrono::high_resolution_clock::now();
  auto build_seconds = std::chrono::duration<double>(build_end - build_start).count();
  printf("Build time: %.3f s\n", build_seconds);

  auto chunk_size = telemetry_query_chunk_size > 0 ? std::min<std::int64_t>(telemetry_query_chunk_size, Nq)
                                                   : static_cast<std::int64_t>(Nq);
  auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, chunk_size, topk);
  auto d_distances = raft::make_device_matrix<float, int64_t>(res, chunk_size, topk);

  auto run_chunk = [&](int64_t offset, int64_t rows) {
    auto query_view = raft::make_device_matrix_view<const float, int64_t>(
      d_queries.data_handle() + offset * dim, rows, dim);
    auto label_view = raft::make_device_vector_view<uint32_t, int64_t>(
      d_query_labels.data_handle() + offset, rows);
    auto neighbor_view = raft::make_device_matrix_view<uint32_t, int64_t>(
      d_neighbors.data_handle(), rows, topk);
    auto distance_view = raft::make_device_matrix_view<float, int64_t>(
      d_distances.data_handle(), rows, topk);
    vecflow::search(res,
                    idx,
                    query_view,
                    label_view,
                    itopk_size,
                    neighbor_view,
                    distance_view);
    raft::resource::sync_stream(res);
  };

  auto run_full_pass = [&](bool capture_samples,
                           const std::chrono::high_resolution_clock::time_point& timed_start,
                           const vecflow_cache_counters_snapshot& base_counters,
                           std::int64_t* queries_completed,
                           std::vector<telemetry_sample>* samples) {
    for (int64_t offset = 0; offset < static_cast<int64_t>(Nq); offset += chunk_size) {
      auto rows = std::min<std::int64_t>(chunk_size, static_cast<int64_t>(Nq) - offset);
      run_chunk(offset, rows);
      if (capture_samples) {
        *queries_completed += rows;
        auto now = std::chrono::high_resolution_clock::now();
        auto counters = subtract_vecflow_cache_counters(capture_vecflow_cache_counters(idx), base_counters);
        samples->push_back(telemetry_sample{
          std::chrono::duration<double>(now - timed_start).count(),
          *queries_completed,
          counters});
      }
    }
  };

  printf("\n=== Warmup ===\n");
  for (int i = 0; i < warmup_runs; ++i) {
    run_full_pass(false,
                  std::chrono::high_resolution_clock::now(),
                  vecflow_cache_counters_snapshot{},
                  nullptr,
                  nullptr);
  }

  printf("\n=== Timed Time-Series Search ===\n");
  auto cache_counters_before = capture_vecflow_cache_counters(idx);
  auto timed_start = std::chrono::high_resolution_clock::now();
  std::int64_t queries_completed = 0;
  std::vector<telemetry_sample> samples;
  samples.reserve(static_cast<std::size_t>(num_runs) *
                  static_cast<std::size_t>((static_cast<std::int64_t>(Nq) + chunk_size - 1) / chunk_size + 1));
  samples.push_back(telemetry_sample{0.0, 0, vecflow_cache_counters_snapshot{}});
  for (int run = 0; run < num_runs; ++run) {
    run_full_pass(true, timed_start, cache_counters_before, &queries_completed, &samples);
  }
  auto timed_end = std::chrono::high_resolution_clock::now();
  auto total_seconds = std::chrono::duration<double>(timed_end - timed_start).count();
  auto cache_counters_after = capture_vecflow_cache_counters(idx);
  auto cache_counter_delta = subtract_vecflow_cache_counters(cache_counters_after, cache_counters_before);
  auto overall_qps = total_seconds > 0.0 ? static_cast<double>(queries_completed) / total_seconds : 0.0;

  printf("Timed queries: %lld\n", static_cast<long long>(queries_completed));
  printf("Total timed seconds: %.3f\n", total_seconds);
  printf("Overall QPS: %.3f\n", overall_qps);

  json samples_json = json::array();
  for (auto const& sample : samples) {
    samples_json.push_back(telemetry_sample_to_json(sample));
  }

  json output = {
    {"config_file", config_file},
    {"num_queries", static_cast<std::int64_t>(Nq)},
    {"num_runs", num_runs},
    {"warmup_runs", warmup_runs},
    {"queries_completed", queries_completed},
    {"itopk", itopk_size},
    {"topk", topk},
    {"telemetry_query_chunk_size", chunk_size},
    {"graph_builder", builder == vecflow::graph_builder_type::TAGORE_CAGRA_COMPAT ? "tagore" : "cagra"},
    {"build_seconds", build_seconds},
    {"timed_seconds", total_seconds},
    {"overall_qps", overall_qps},
    {"cache_counters_total", vecflow_cache_counters_to_json(cache_counter_delta)},
    {"samples", samples_json},
    {"diagnostic_note",
     "Chunked single-label diagnostic path. Use for per-second QPS/hit-rate analysis, not as a formal apples-to-apples throughput benchmark."}
  };

  auto output_path = std::filesystem::path(output_json_file);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream out(output_path);
  out << std::setw(2) << output << std::endl;
  printf("Wrote time-series diagnostic JSON to %s\n", output_json_file.c_str());
  return 0;
}
