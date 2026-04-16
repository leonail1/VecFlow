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

#ifdef CUVS_VECFLOW_TAGORE_ENABLED

#include <raft/core/error.hpp>
#include <rmm/device_uvector.hpp>

#include <Tagore_src.cuh>

#include "tagore_device_helpers.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cuvs::neighbors::vecflow::detail::tagore {

inline constexpr unsigned k_candidate_degree = K_SIZE;

__global__ inline void convert_float_to_half_kernel(const float* input,
                                                    half* output,
                                                    std::size_t size,
                                                    float scale)
{
  auto idx    = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (auto i = idx; i < size; i += stride) {
    output[i] = __float2half(input[i] * scale);
  }
}

inline void validate_build_inputs(std::int64_t points,
                                  std::int64_t dim,
                                  unsigned graph_degree,
                                  const std::string& label_name)
{
  if (points <= 0) {
    throw std::invalid_argument("Tagore build requires at least one point for label " + label_name);
  }
  if (dim <= 0) {
    throw std::invalid_argument("Tagore build requires a positive dimension for label " +
                                label_name);
  }
  if ((dim % 32) != 0) {
    throw std::invalid_argument("Tagore build currently requires dimension to be a multiple of 32, got " +
                                std::to_string(dim) + " for label " + label_name);
  }
  if (dim > DIM_SIZE) {
    throw std::invalid_argument("Tagore build currently supports dimensions up to " +
                                std::to_string(DIM_SIZE) + ", got " + std::to_string(dim) +
                                " for label " + label_name);
  }
  if (graph_degree > FINAL_DEGREE_SIZE) {
    throw std::invalid_argument("Tagore build currently supports graph_degree <= " +
                                std::to_string(FINAL_DEGREE_SIZE) + ", got " +
                                std::to_string(graph_degree) + " for label " + label_name);
  }
}

inline auto compute_normalization_factor(raft::resources const& res,
                                         raft::device_matrix_view<const float, int64_t> dataset)
  -> float
{
  auto dim    = static_cast<unsigned>(dataset.extent(1));
  auto stream = raft::resource::get_cuda_stream(res);

  std::vector<float> first_row(dim);
  raft::copy(first_row.data(), dataset.data_handle(), dim, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  float tmp_ave = 0.0f;
  for (auto value : first_row) {
    tmp_ave += std::abs(value);
  }
  tmp_ave /= std::max(1u, dim);

  float norm_factor = 1.0f;
  if (tmp_ave > 0.0f) {
    while (tmp_ave < 0.5f) {
      tmp_ave *= 10.0f;
      norm_factor *= 10.0f;
    }
    while (tmp_ave > 5.0f) {
      tmp_ave /= 10.0f;
      norm_factor /= 10.0f;
    }
  }
  return norm_factor;
}

inline void build_cagra_compatible_graph(raft::resources const& res,
                                         raft::device_matrix_view<const float, int64_t> dataset,
                                         unsigned graph_degree,
                                         unsigned iterations,
                                         raft::host_matrix_view<uint32_t, int64_t> output)
{
  auto points_num = static_cast<unsigned>(dataset.extent(0));
  auto dim        = static_cast<unsigned>(dataset.extent(1));
  auto stream     = raft::resource::get_cuda_stream(res);
  auto total_size = static_cast<std::size_t>(points_num) * dim;
  auto norm_factor = compute_normalization_factor(res, dataset);

  rmm::device_uvector<half> data_half(total_size, stream);
  rmm::device_uvector<unsigned> graph_dev(static_cast<std::size_t>(points_num) * k_candidate_degree,
                                          stream);
  rmm::device_uvector<unsigned> reverse_graph_dev(
    static_cast<std::size_t>(points_num) * RESERVENUM, stream);
  rmm::device_uvector<float> nei_distance(
    static_cast<std::size_t>(points_num) * k_candidate_degree, stream);
  rmm::device_uvector<float> reverse_distance(
    static_cast<std::size_t>(points_num) * RESERVENUM, stream);
  rmm::device_uvector<bool> nei_visit(
    static_cast<std::size_t>(points_num) * k_candidate_degree, stream);
  rmm::device_uvector<unsigned> reverse_num(points_num, stream);
  rmm::device_uvector<unsigned> reverse_num_old(points_num, stream);
  rmm::device_uvector<unsigned> new_num_global(points_num, stream);
  rmm::device_uvector<unsigned> old_num_global(points_num, stream);
  rmm::device_uvector<unsigned> hybrid_list(
    static_cast<std::size_t>(points_num) * SAMPLE * 4, stream);
  rmm::device_uvector<float> data_power(points_num, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(reverse_num.data(), 0, reverse_num.size() * sizeof(unsigned), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(reverse_num_old.data(), 0, reverse_num_old.size() * sizeof(unsigned), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    new_num_global.data(), 0, new_num_global.size() * sizeof(unsigned), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    old_num_global.data(), 0, old_num_global.size() * sizeof(unsigned), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(data_power.data(), 0, data_power.size() * sizeof(float), stream));

  auto threads = 256u;
  auto blocks  = static_cast<unsigned>((total_size + threads - 1) / threads);
  convert_float_to_half_kernel<<<blocks, threads, 0, stream>>>(
    dataset.data_handle(), data_half.data(), total_size, norm_factor);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  dim3 graph_grid(points_num, 1, 1);
  dim3 phase2_block(32, MAX_P / 32, 1);
  dim3 phase1_block(32, 16, 1);
  dim3 merge_block(32, 3, 1);
  dim3 prune_block(32, 4, 1);

  auto phase1_iters = std::max(1u, iterations / 2u);
  auto phase2_iters = std::max(1u, iterations - phase1_iters);

  initialize_graph<<<points_num, 32, 0, stream>>>(
    graph_dev.data(), points_num, nei_distance.data(), nei_visit.data(), k_candidate_degree);
  cal_power<<<points_num, dim, 0, stream>>>(data_half.data(), data_power.data(), dim, dim);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  for (unsigned it = 0; it < phase1_iters; ++it) {
    nn_descent_opt_sample<<<points_num, 32, 0, stream>>>(graph_dev.data(),
                                                          reverse_graph_dev.data(),
                                                          nei_visit.data(),
                                                          reverse_num.data(),
                                                          reverse_num_old.data(),
                                                          new_num_global.data(),
                                                          old_num_global.data(),
                                                          hybrid_list.data(),
                                                          k_candidate_degree);
    nn_descent_opt_reverse_sample<<<points_num, 32, 0, stream>>>(graph_dev.data(),
                                                                  reverse_graph_dev.data(),
                                                                  reverse_num.data(),
                                                                  reverse_num_old.data(),
                                                                  new_num_global.data(),
                                                                  old_num_global.data(),
                                                                  hybrid_list.data(),
                                                                  k_candidate_degree);
    reset_reverse_new_old_num<<<1000, 1024, 0, stream>>>(
      reverse_num.data(), reverse_num_old.data(), points_num);
    nn_descent_opt_cal<<<graph_grid, phase1_block, 0, stream>>>(graph_dev.data(),
                                                                 reverse_graph_dev.data(),
                                                                 data_half.data(),
                                                                 data_power.data(),
                                                                 it,
                                                                 reverse_distance.data(),
                                                                 reverse_num.data(),
                                                                 new_num_global.data(),
                                                                 old_num_global.data(),
                                                                 hybrid_list.data(),
                                                                 dim,
                                                                 k_candidate_degree);
    nn_descent_opt_merge<<<graph_grid, merge_block, 0, stream>>>(graph_dev.data(),
                                                                  reverse_graph_dev.data(),
                                                                  it,
                                                                  phase1_iters,
                                                                  nei_distance.data(),
                                                                  reverse_distance.data(),
                                                                  nei_visit.data(),
                                                                  reverse_num.data(),
                                                                  k_candidate_degree);
    reset_reverse_num<<<1000, 1024, 0, stream>>>(reverse_num.data(), points_num);
  }

  do_reverse_graph<<<points_num, 32, 0, stream>>>(graph_dev.data(),
                                                   reverse_graph_dev.data(),
                                                   nei_distance.data(),
                                                   reverse_distance.data(),
                                                   reverse_num.data(),
                                                   k_candidate_degree);
  reset_visit_reversenum<<<points_num, 32, 0, stream>>>(reverse_graph_dev.data(),
                                                         nei_visit.data(),
                                                         reverse_num.data(),
                                                         points_num,
                                                         k_candidate_degree);

  for (unsigned it = 0; it < phase2_iters; ++it) {
    sample_kernel6<<<graph_grid, phase2_block, 0, stream>>>(graph_dev.data(),
                                                             reverse_graph_dev.data(),
                                                             it,
                                                             data_half.data(),
                                                             points_num,
                                                             phase2_iters,
                                                             nei_distance.data(),
                                                             reverse_distance.data(),
                                                             nei_visit.data(),
                                                             reverse_num.data(),
                                                             dim,
                                                             k_candidate_degree);
    if (it + 1 < phase2_iters) {
      reset_reverse_num<<<1000, 1024, 0, stream>>>(reverse_num.data(), points_num);
    }
  }

  merge_reverse_plus<<<graph_grid, merge_block, 0, stream>>>(graph_dev.data(),
                                                              reverse_graph_dev.data(),
                                                              nei_distance.data(),
                                                              reverse_distance.data(),
                                                              reverse_num.data(),
                                                              k_candidate_degree);
  reset_reverse_num<<<1000, 1024, 0, stream>>>(reverse_num.data(), points_num);

  select_1hop_cagra<<<graph_grid, prune_block, 0, stream>>>(graph_dev.data(),
                                                             graph_degree,
                                                             reverse_graph_dev.data(),
                                                             k_candidate_degree,
                                                             reverse_num.data(),
                                                             hybrid_list.data());
  filter_reverse_1hop<<<graph_grid, prune_block, 0, stream>>>(graph_dev.data(),
                                                               reverse_graph_dev.data(),
                                                               graph_degree,
                                                               k_candidate_degree,
                                                               reverse_num.data(),
                                                               hybrid_list.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  RAFT_CUDA_TRY(cudaMemcpy(output.data_handle(),
                           graph_dev.data(),
                           static_cast<std::size_t>(points_num) * graph_degree * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost));
}

}  // namespace cuvs::neighbors::vecflow::detail::tagore

#endif
