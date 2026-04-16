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

#include "vecflow_common.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace cuvs::neighbors::vecflow::detail::multi_gpu {

struct label_assignment {
  std::unordered_map<uint32_t, int> label_to_worker;
  std::vector<std::vector<uint32_t>> worker_labels;
  std::vector<double> worker_loads;
};

template <typename data_t>
inline auto copy_device_matrix_to_host(shared_resources::configured_raft_resources& res,
                                       raft::device_matrix_view<const data_t, int64_t> dataset)
  -> std::vector<data_t>
{
  auto total_values = dataset.extent(0) * dataset.extent(1);
  std::vector<data_t> host_values(static_cast<std::size_t>(total_values));
  raft::copy(host_values.data(),
             dataset.data_handle(),
             total_values,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return host_values;
}

inline auto compute_label_counts(const std::vector<std::vector<int>>& data_label_vecs)
  -> std::vector<int64_t>
{
  auto max_label = -1;
  for (auto const& labels : data_label_vecs) {
    for (auto label : labels) {
      if (label > max_label) { max_label = label; }
    }
  }

  std::vector<int64_t> label_counts(max_label >= 0 ? static_cast<std::size_t>(max_label + 1) : 0,
                                    0);
  for (auto const& labels : data_label_vecs) {
    for (auto label : labels) {
      if (label >= 0) { label_counts[static_cast<std::size_t>(label)] += 1; }
    }
  }
  return label_counts;
}

inline auto assign_labels_to_workers(const std::vector<int64_t>& label_counts,
                                     const cuvs::neighbors::vecflow::multi_gpu_params& params)
  -> label_assignment
{
  if (params.device_ids.empty()) {
    throw std::invalid_argument("build_multi_gpu requires at least one device id");
  }

  struct weighted_label {
    uint32_t label = 0;
    double weight  = 0.0;
  };

  std::vector<weighted_label> weighted_labels;
  weighted_labels.reserve(label_counts.size());
  for (std::size_t label = 0; label < label_counts.size(); ++label) {
    auto count = label_counts[label];
    if (count <= 0) { continue; }
    auto query_weight =
      label < params.label_query_weights.size() ? params.label_query_weights[label] : 0.0;
    auto weight = params.label_routing_data_weight * static_cast<double>(count) +
                  params.label_routing_query_weight * query_weight;
    weighted_labels.push_back(weighted_label{static_cast<uint32_t>(label), weight});
  }

  std::sort(weighted_labels.begin(),
            weighted_labels.end(),
            [](weighted_label const& lhs, weighted_label const& rhs) {
              if (lhs.weight != rhs.weight) { return lhs.weight > rhs.weight; }
              return lhs.label < rhs.label;
            });

  label_assignment assignment;
  assignment.worker_labels.resize(params.device_ids.size());
  assignment.worker_loads.assign(params.device_ids.size(), 0.0);

  for (auto const& item : weighted_labels) {
    auto worker_it =
      std::min_element(assignment.worker_loads.begin(), assignment.worker_loads.end());
    auto worker = static_cast<int>(std::distance(assignment.worker_loads.begin(), worker_it));
    assignment.label_to_worker[item.label] = worker;
    assignment.worker_labels[static_cast<std::size_t>(worker)].push_back(item.label);
    *worker_it += item.weight;
  }

  for (auto& labels : assignment.worker_labels) {
    std::sort(labels.begin(), labels.end());
  }
  return assignment;
}

inline auto filter_labels_by_owner(const std::vector<std::vector<int>>& data_label_vecs,
                                   const std::vector<uint32_t>& owned_labels)
  -> std::vector<std::vector<int>>
{
  std::unordered_set<uint32_t> owned(owned_labels.begin(), owned_labels.end());
  std::vector<std::vector<int>> filtered = data_label_vecs;
  for (auto& labels : filtered) {
    labels.erase(std::remove_if(labels.begin(),
                                labels.end(),
                                [&](int label) {
                                  return label < 0 ||
                                         owned.find(static_cast<uint32_t>(label)) == owned.end();
                                }),
                 labels.end());
  }
  return filtered;
}

template <typename data_t>
inline void gather_host_query_subset(const std::vector<data_t>& host_queries,
                                     int64_t dim,
                                     const std::vector<uint32_t>& host_query_labels,
                                     const std::vector<int64_t>& query_ids,
                                     std::vector<data_t>* subset_queries,
                                     std::vector<uint32_t>* subset_labels)
{
  subset_queries->resize(static_cast<std::size_t>(query_ids.size() * dim));
  subset_labels->resize(query_ids.size());

  for (std::size_t i = 0; i < query_ids.size(); ++i) {
    auto query_id = query_ids[i];
    std::copy_n(host_queries.data() + query_id * dim,
                dim,
                subset_queries->data() + static_cast<int64_t>(i) * dim);
    (*subset_labels)[i] = host_query_labels[static_cast<std::size_t>(query_id)];
  }
}

inline auto route_queries_to_workers(const std::vector<uint32_t>& host_query_labels,
                                     const std::unordered_map<uint32_t, int>& label_to_worker,
                                     std::size_t worker_count)
  -> std::vector<std::vector<int64_t>>
{
  std::vector<std::vector<int64_t>> worker_query_ids(worker_count);
  for (std::size_t query_id = 0; query_id < host_query_labels.size(); ++query_id) {
    auto label = host_query_labels[query_id];
    if (label == std::numeric_limits<uint32_t>::max()) { continue; }
    auto it = label_to_worker.find(label);
    if (it == label_to_worker.end()) {
      throw std::runtime_error("search_multi_gpu encountered an unmapped label " +
                               std::to_string(label));
    }
    worker_query_ids[static_cast<std::size_t>(it->second)].push_back(
      static_cast<int64_t>(query_id));
  }
  return worker_query_ids;
}

inline auto make_worker_cache_filename(const std::string& filename, int device_id) -> std::string
{
  if (filename.empty()) { return std::string{}; }
  return append_suffix_to_filename(filename, "_mg_d" + std::to_string(device_id));
}

}  // namespace cuvs::neighbors::vecflow::detail::multi_gpu
