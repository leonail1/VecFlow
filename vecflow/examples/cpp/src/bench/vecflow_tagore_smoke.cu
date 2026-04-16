#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <cmath>
#include <cstdint>
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
                               std::int64_t query_count) -> std::vector<float>
{
  std::vector<float> queries(static_cast<std::size_t>(query_count * dim));
  for (std::int64_t i = 0; i < query_count; ++i) {
    auto source_row = (i * 7) % rows;
    for (std::int64_t j = 0; j < dim; ++j) {
      queries[static_cast<std::size_t>(i * dim + j)] =
        data[static_cast<std::size_t>(source_row * dim + j)];
    }
  }
  return queries;
}

}  // namespace

int main()
{
  constexpr std::int64_t rows        = 256;
  constexpr std::int64_t dim         = 192;
  constexpr std::int64_t query_count = 8;
  constexpr int graph_degree         = 16;
  constexpr int specificity_threshold = 1;
  constexpr int itopk_size           = 32;
  constexpr int topk                 = 10;

  auto h_data    = make_synthetic_dataset(rows, dim);
  auto h_queries = make_queries_from_dataset(h_data, rows, dim, query_count);

  std::vector<std::vector<int>> data_label_vecs(static_cast<std::size_t>(rows), std::vector<int>{0});
  std::vector<uint32_t> h_query_labels(static_cast<std::size_t>(query_count), 0);

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
    "",
    true,
    cuvs::neighbors::vecflow::graph_builder_type::TAGORE_CAGRA_COMPAT,
    6);

  auto stats = cuvs::neighbors::vecflow::storage_stats(index);
  if (stats.graph.total_labels == 0 || stats.graph.hbm.labels == 0) {
    std::cerr << "Smoke test failed: storage_stats reported no graph labels in HBM\n";
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
  }

  std::cout << "Tagore 192D smoke test passed. First query first neighbor: "
            << h_neighbors[0] << ", graph labels in HBM: " << stats.graph.hbm.labels << "\n";
  return 0;
}
