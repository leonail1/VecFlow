#include <cuvs/neighbors/vecflow.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <cuvs/neighbors/shared_resources.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

auto make_synthetic_dataset(std::int64_t rows, std::int64_t dim) -> std::vector<float>
{
  std::vector<float> data(static_cast<std::size_t>(rows * dim));
  for (std::int64_t i = 0; i < rows; ++i) {
    auto label_bias = static_cast<float>(i / (rows / 4)) * 8.0f;
    for (std::int64_t j = 0; j < dim; ++j) {
      auto value = static_cast<float>(((i * 17 + j * 13) % 97) - 48) / 32.0f + label_bias;
      data[static_cast<std::size_t>(i * dim + j)] = value;
    }
  }
  return data;
}

}  // namespace

int main()
{
  int device_count = 0;
  RAFT_CUDA_TRY(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    std::cerr << "Multi-GPU smoke test requires at least one CUDA device\n";
    return 1;
  }

  constexpr std::int64_t rows         = 512;
  constexpr std::int64_t dim          = 128;
  constexpr std::int64_t query_count  = 16;
  constexpr int graph_degree          = 16;
  constexpr int specificity_threshold = 1;
  constexpr int itopk_size            = 32;
  constexpr int topk                  = 10;

  auto h_data = make_synthetic_dataset(rows, dim);
  std::vector<std::vector<int>> data_label_vecs(static_cast<std::size_t>(rows));
  for (std::int64_t row = 0; row < rows; ++row) {
    data_label_vecs[static_cast<std::size_t>(row)] = {
      static_cast<int>(row / (rows / 4))
    };
  }

  std::vector<float> h_queries(static_cast<std::size_t>(query_count * dim));
  std::vector<uint32_t> h_query_labels(static_cast<std::size_t>(query_count));
  for (std::int64_t i = 0; i < query_count; ++i) {
    auto source_row = (i * 29) % rows;
    auto label      = static_cast<uint32_t>(source_row / (rows / 4));
    std::copy_n(h_data.data() + source_row * dim,
                dim,
                h_queries.data() + static_cast<std::size_t>(i * dim));
    h_query_labels[static_cast<std::size_t>(i)] = label;
  }

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

  cuvs::neighbors::vecflow::multi_gpu_params mg_params;
  for (int device_id = 0; device_id < std::min(device_count, 2); ++device_id) {
    mg_params.device_ids.push_back(device_id);
  }

  auto mg_index = cuvs::neighbors::vecflow::build_multi_gpu(
    res,
    raft::make_const_mdspan(d_dataset.view()),
    data_label_vecs,
    graph_degree,
    specificity_threshold,
    mg_params,
    "",
    "",
    true,
    cuvs::neighbors::vecflow::graph_builder_type::TAGORE_CAGRA_COMPAT,
    6);

  auto mg_stats = cuvs::neighbors::vecflow::storage_stats(mg_index);
  if (mg_stats.worker_storage.size() != mg_index.workers.size() || mg_stats.mapped_labels != 4) {
    std::cerr << "Multi-GPU API smoke test failed: storage_stats returned inconsistent worker state\n";
    return 1;
  }

  cuvs::neighbors::vecflow::search_multi_gpu(res,
                                             mg_index,
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
      std::cerr << "Multi-GPU API smoke test failed: query " << i
                << " returned no valid neighbors\n";
      return 1;
    }
  }

  std::cout << "VecFlow multi-GPU C++ API smoke test passed on " << mg_params.device_ids.size()
            << " device(s). First query first neighbor: " << h_neighbors[0]
            << ", mapped labels: " << mg_stats.mapped_labels << "\n";
  return 0;
}
