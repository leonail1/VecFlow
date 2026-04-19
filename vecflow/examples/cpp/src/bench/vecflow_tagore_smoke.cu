#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <vector>

namespace {

auto make_synthetic_dataset(std::int64_t rows, std::int64_t dim) -> std::vector<float>
{
  std::vector<float> data(static_cast<std::size_t>(rows * dim));
  for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t j = 0; j < dim; ++j) {
      auto value = static_cast<float>(((i * 17 + j * 13) % 97) - 48) / 32.0f;
      data[static_cast<std::size_t>(i * dim + j)] = value;
    }
  }
  return data;
}

auto make_queries_from_dataset(const std::vector<float>& data,
                               std::int64_t rows,
                               std::int64_t dim,
                               const std::vector<std::int64_t>& source_rows) -> std::vector<float>
{
  std::vector<float> queries(static_cast<std::size_t>(source_rows.size() * dim));
  for (std::size_t i = 0; i < source_rows.size(); ++i) {
    auto source_row = source_rows[i] % rows;
    for (std::int64_t j = 0; j < dim; ++j) {
      queries[static_cast<std::size_t>(static_cast<std::int64_t>(i) * dim + j)] =
        data[static_cast<std::size_t>(source_row * dim + j)];
    }
  }
  return queries;
}

auto append_suffix_to_filename(const std::string& filename, const std::string& suffix)
  -> std::string
{
  auto path = std::filesystem::path(filename);
  auto stem = path.stem().string();
  auto ext = path.extension().string();
  auto parent = path.parent_path();
  return (parent / (stem + suffix + ext)).string();
}

}  // namespace

int main()
{
  constexpr std::int64_t rows        = 256;
  constexpr std::int64_t dim         = 192;
  constexpr std::int64_t query_count = 8;
  constexpr int graph_degree         = 16;
  constexpr int specificity_threshold = 16;
  constexpr int itopk_size           = 32;
  constexpr int topk                 = 10;

  auto h_data    = make_synthetic_dataset(rows, dim);
  std::vector<std::vector<int>> data_label_vecs(static_cast<std::size_t>(rows));
  for (std::int64_t i = 0; i < 200; ++i) {
    data_label_vecs[static_cast<std::size_t>(i)] = {0};
  }
  std::int64_t bfs_row = 200;
  for (int label = 1; label <= 7; ++label) {
    for (int j = 0; j < 8; ++j) {
      data_label_vecs[static_cast<std::size_t>(bfs_row++)] = {label};
    }
  }

  std::vector<std::int64_t> query_source_rows{0, 7, 14, 21, 200, 208, 216, 224};
  auto h_queries = make_queries_from_dataset(h_data, rows, dim, query_source_rows);
  std::vector<uint32_t> h_query_labels{0, 0, 0, 0, 1, 2, 3, 4};

  auto bfs_cache_path =
    (std::filesystem::temp_directory_path() / "vecflow_bfs_tiered_smoke.ibin").string();
  auto bfs_dataset_cache_path = append_suffix_to_filename(bfs_cache_path, "_dataset");
  std::filesystem::remove(bfs_cache_path);
  std::filesystem::remove(bfs_dataset_cache_path);

  cuvs::neighbors::vecflow::runtime_config runtime_config;
  runtime_config.cascade_eviction             = true;
  runtime_config.enable_bfs_tiered_cache      = true;
  runtime_config.use_phoenix_label_load       = false;
  runtime_config.use_phoenix_graph_load       = false;
  runtime_config.bfs_hbm_cache_bytes          = static_cast<std::size_t>(26000);
  runtime_config.bfs_dram_cache_bytes         = static_cast<std::size_t>(13000);
  runtime_config.bfs_prefetch_max_bytes       = static_cast<std::size_t>(13000);
  runtime_config.bfs_rebalance_interval_queries = static_cast<std::size_t>(4);
  cuvs::neighbors::vecflow::scoped_runtime_config runtime_config_scope(runtime_config);

  shared_resources::configured_raft_resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  auto d_dataset = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, query_count, dim);
  auto d_query_labels = raft::make_device_vector<uint32_t, int64_t>(res, query_count);
  auto d_neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, query_count, topk);
  auto d_distances = raft::make_device_matrix<float, int64_t>(res, query_count, topk);

  raft::copy(d_dataset.data_handle(), h_data.data(), rows * dim, stream);
  raft::copy(d_queries.data_handle(), h_queries.data(), query_count * dim, stream);
  raft::copy(d_query_labels.data_handle(), h_query_labels.data(), query_count, stream);
  raft::resource::sync_stream(res);

  auto index = cuvs::neighbors::vecflow::build(
    res,
    raft::make_const_mdspan(d_dataset.view()),
    data_label_vecs,
    graph_degree,
    specificity_threshold,
    "",
    bfs_cache_path,
    true,
    cuvs::neighbors::vecflow::graph_builder_type::TAGORE_CAGRA_COMPAT,
    6);

  auto stats = cuvs::neighbors::vecflow::storage_stats(index);
  if (stats.graph.total_labels == 0 || stats.graph.hbm.labels == 0) {
    std::cerr << "Smoke test failed: storage_stats reported no graph labels in HBM\n";
    return 1;
  }
  if (!index.bfs_tiered_cache_enabled || !std::filesystem::exists(bfs_dataset_cache_path)) {
    std::cerr << "Smoke test failed: tiered BFS cache was not enabled or cache file missing\n";
    return 1;
  }
  if (index.bfs_cache == nullptr || index.bfs_cache->hbm_entries.empty() ||
      index.bfs_cache->dram_entries.empty()) {
    std::cerr << "Smoke test failed: BFS cache did not populate HBM and DRAM tiers\n";
    return 1;
  }

  cuvs::neighbors::vecflow::search(res,
                                   index,
                                   raft::make_const_mdspan(d_queries.view()),
                                   d_query_labels.view(),
                                   itopk_size,
                                   d_neighbors.view(),
                                   d_distances.view());
  raft::resource::sync_stream(res);

  std::vector<uint32_t> h_neighbors(static_cast<std::size_t>(query_count * topk));
  raft::copy(h_neighbors.data(), d_neighbors.data_handle(), query_count * topk, stream);
  raft::resource::sync_stream(res);

  for (std::int64_t i = 0; i < query_count; ++i) {
    auto first_neighbor = h_neighbors[static_cast<std::size_t>(i * topk)];
    if (first_neighbor == std::numeric_limits<uint32_t>::max()) {
      std::cerr << "Smoke test failed: query " << i << " returned no valid neighbors\n";
      return 1;
    }
    if (data_label_vecs[first_neighbor].empty() ||
        static_cast<uint32_t>(data_label_vecs[first_neighbor][0]) != h_query_labels[static_cast<std::size_t>(i)]) {
      std::cerr << "Smoke test failed: query " << i
                << " returned a neighbor with the wrong label\n";
      return 1;
    }
  }

  std::cout << "Tagore/BFS tiered smoke test passed. First query first neighbor: "
            << h_neighbors[0] << ", graph labels in HBM: " << stats.graph.hbm.labels
            << ", bfs HBM labels: " << index.bfs_cache->hbm_entries.size()
            << ", bfs DRAM labels: " << index.bfs_cache->dram_entries.size() << "\n";
  return 0;
}
