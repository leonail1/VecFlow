#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <vector>
#include <fcntl.h>

using json = nlohmann::json;
namespace fs = std::filesystem;

struct query_chunk {
  int64_t offset = 0;
  int64_t size = 0;
};

struct worker_process {
  int device_id = 0;
  int64_t query_offset = 0;
  int64_t query_count = 0;
  std::vector<int64_t> query_ids;
  std::vector<uint32_t> labels;
  fs::path query_id_list_path;
  fs::path allowed_labels_path;
  fs::path config_path;
  fs::path output_path;
  fs::path log_path;
  pid_t pid = -1;
};

struct result_aggregate {
  double weighted_recall_sum = 0.0;
  double max_build_seconds = 0.0;
  double max_search_seconds = 0.0;
  int64_t num_queries = 0;
};

struct label_routing_plan {
  std::vector<std::vector<int64_t>> worker_query_ids;
  std::unordered_map<uint32_t, int> label_to_worker;
  std::vector<double> worker_loads;
};

void read_fbin_header(const std::string& path, uint32_t* n_rows, uint32_t* dim);
std::string trim_copy(std::string value);
std::vector<int64_t> read_data_label_counts(const std::string& path);

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

std::string sanitize_component(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return std::isalnum(ch) ? static_cast<char>(ch) : '_';
  });
  return value;
}

std::string append_suffix_to_filename(const std::string& filename, const std::string& suffix)
{
  auto path = fs::path(filename);
  auto stem = path.stem().string();
  auto ext = path.extension().string();
  auto parent = path.parent_path();
  return (parent / (stem + suffix + ext)).string();
}

std::string choose_worker_cache_filename(const json& config,
                                         const char* key,
                                         const std::string& default_value,
                                         const std::string& suffix,
                                         bool allow_reuse_existing = true)
{
  auto original = config.value(key, default_value);
  auto force_rebuild = config.value("force_rebuild", false);
  auto data_dir = fs::path(config["data_dir"].get<std::string>());
  if (allow_reuse_existing && !force_rebuild && !original.empty() && fs::exists(data_dir / original)) {
    return original;
  }
  return append_suffix_to_filename(original, suffix);
}

std::string primary_graph_cache_filename(const json& config, const std::string& worker_algorithm)
{
  if (worker_algorithm == "vecflow_tagore") {
    return config.value("ivf_graph_tagore_fname", std::string("ivf_graph_tagore.bin"));
  }
  return config.value("ivf_graph_fname", std::string{});
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
      auto trimmed = trim_copy(token);
      if (trimmed.empty()) { continue; }
      labels.insert(std::stoi(trimmed));
    }
  }
  return labels;
}

std::pair<int64_t, int64_t> read_ibin_header(const fs::path& path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to open ibin file: " + path.string());
  }

  int64_t rows = 0;
  int64_t cols = 0;
  input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
  if (!input) {
    throw std::runtime_error("Unable to read ibin header from: " + path.string());
  }
  return {rows, cols};
}

int64_t expected_cagra_rows(const json& config)
{
  auto data_dir = fs::path(config["data_dir"].get<std::string>());
  auto counts =
    read_data_label_counts((data_dir / config["data_label_fname"].get<std::string>()).string());

  auto allowed_labels_file = config.value("allowed_labels_file", std::string{});
  if (!allowed_labels_file.empty()) {
    auto allowed = read_allowed_labels_file(allowed_labels_file);
    for (std::size_t label = 0; label < counts.size(); ++label) {
      if (allowed.count(static_cast<int>(label)) == 0) { counts[label] = 0; }
    }
  }

  auto specificity_threshold = config.value("spec_threshold", 0);
  int64_t rows = 0;
  for (auto count : counts) {
    if (count > specificity_threshold) { rows += count; }
  }
  return rows;
}

bool matches_ibin_shape(const fs::path& path, int64_t expected_rows, int64_t expected_cols)
{
  if (!fs::exists(path)) { return false; }
  auto [rows, cols] = read_ibin_header(path);
  return rows == expected_rows && cols == expected_cols;
}

bool should_prebuild_shared_cache(const json& config, const std::string& worker_algorithm)
{
  if (config.value("force_rebuild", false)) { return true; }
  auto data_dir = fs::path(config["data_dir"].get<std::string>());
  auto graph_file = primary_graph_cache_filename(config, worker_algorithm);
  if (graph_file.empty()) { return false; }

  auto graph_path = data_dir / graph_file;
  if (!fs::exists(graph_path)) { return true; }

  auto data_path = data_dir / config["data_fname"].get<std::string>();
  uint32_t n_rows = 0;
  uint32_t dim = 0;
  read_fbin_header(data_path.string(), &n_rows, &dim);

  auto expected_rows = expected_cagra_rows(config);
  try {
    auto [graph_rows, graph_cols] = read_ibin_header(graph_path);
    if (graph_rows != expected_rows || graph_cols <= 0) { return true; }
  } catch (...) {
    return true;
  }

  if (config.value("use_phoenix_label_load", false)) {
    auto dataset_path = data_dir / append_suffix_to_filename(graph_file, "_dataset");
    try {
      if (!matches_ibin_shape(dataset_path, expected_rows, dim)) { return true; }
    } catch (...) {
      return true;
    }
  }

  return false;
}

fs::path get_self_executable_path()
{
  std::array<char, PATH_MAX> buffer{};
  auto len = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (len < 0) {
    throw std::runtime_error("Unable to resolve /proc/self/exe: " + std::string(std::strerror(errno)));
  }
  buffer[static_cast<size_t>(len)] = '\0';
  return fs::path(buffer.data());
}

void read_fbin_header(const std::string& path, uint32_t* n_rows, uint32_t* dim)
{
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to open query file: " + path);
  }
  input.read(reinterpret_cast<char*>(n_rows), sizeof(uint32_t));
  input.read(reinterpret_cast<char*>(dim), sizeof(uint32_t));
  if (!input) { throw std::runtime_error("Unable to read fbin header from: " + path); }
}

bool is_text_label_file(const std::string& path)
{
  return path.find(".txt") != std::string::npos;
}

std::string trim_copy(std::string value)
{
  auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) { return std::string{}; }
  auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<uint32_t> read_first_query_labels(const std::string& path)
{
  if (is_text_label_file(path)) {
    std::ifstream input(path);
    if (!input.is_open()) {
      throw std::runtime_error("Unable to open query label file: " + path);
    }

    std::vector<uint32_t> first_labels;
    std::string line;
    while (std::getline(input, line)) {
      auto first = std::numeric_limits<uint32_t>::max();
      std::stringstream ss(line);
      std::string token;
      if (std::getline(ss, token, ',')) {
        auto trimmed = trim_copy(token);
        if (!trimmed.empty() && trimmed != "-1") {
          first = static_cast<uint32_t>(std::stoul(trimmed));
        }
      }
      first_labels.push_back(first);
    }
    return first_labels;
  }

  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to open query label file: " + path);
  }
  std::array<int64_t, 3> sizes{};
  input.read(reinterpret_cast<char*>(sizes.data()), sizeof(int64_t) * sizes.size());
  auto n_rows = sizes[0];
  auto nnz = sizes[2];
  std::vector<int64_t> indptr(static_cast<size_t>(n_rows + 1));
  input.read(reinterpret_cast<char*>(indptr.data()), sizeof(int64_t) * indptr.size());
  std::vector<int> indices(static_cast<size_t>(nnz));
  input.read(reinterpret_cast<char*>(indices.data()), sizeof(int) * indices.size());
  if (!input) {
    throw std::runtime_error("Unable to read spmat labels from: " + path);
  }

  std::vector<uint32_t> first_labels(static_cast<size_t>(n_rows), std::numeric_limits<uint32_t>::max());
  for (int64_t row = 0; row < n_rows; ++row) {
    if (indptr[row] < indptr[row + 1]) {
      first_labels[static_cast<size_t>(row)] = static_cast<uint32_t>(indices[indptr[row]]);
    }
  }
  return first_labels;
}

std::vector<int64_t> read_data_label_counts(const std::string& path)
{
  std::vector<int64_t> counts;
  if (is_text_label_file(path)) {
    std::ifstream input(path);
    if (!input.is_open()) {
      throw std::runtime_error("Unable to open data label file: " + path);
    }

    std::string line;
    while (std::getline(input, line)) {
      std::stringstream ss(line);
      std::string token;
      while (std::getline(ss, token, ',')) {
        auto trimmed = trim_copy(token);
        if (trimmed.empty() || trimmed == "-1") { continue; }
        auto label = static_cast<size_t>(std::stoul(trimmed));
        if (label >= counts.size()) { counts.resize(label + 1, 0); }
        counts[label] += 1;
      }
    }
    return counts;
  }

  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to open data label file: " + path);
  }
  std::array<int64_t, 3> sizes{};
  input.read(reinterpret_cast<char*>(sizes.data()), sizeof(int64_t) * sizes.size());
  auto n_cols = sizes[1];
  auto nnz = sizes[2];
  std::vector<int64_t> indptr(static_cast<size_t>(sizes[0] + 1));
  input.read(reinterpret_cast<char*>(indptr.data()), sizeof(int64_t) * indptr.size());
  std::vector<int> indices(static_cast<size_t>(nnz));
  input.read(reinterpret_cast<char*>(indices.data()), sizeof(int) * indices.size());
  if (!input) {
    throw std::runtime_error("Unable to read spmat labels from: " + path);
  }

  counts.resize(static_cast<size_t>(n_cols), 0);
  for (auto label : indices) {
    if (label >= 0 && static_cast<size_t>(label) < counts.size()) {
      counts[static_cast<size_t>(label)] += 1;
    }
  }
  return counts;
}

void write_query_id_list(const fs::path& path, const std::vector<int64_t>& query_ids)
{
  if (!path.parent_path().empty()) { fs::create_directories(path.parent_path()); }
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write query id list: " + path.string());
  }
  for (auto query_id : query_ids) {
    output << query_id << '\n';
  }
}

void write_allowed_labels(const fs::path& path, const std::vector<uint32_t>& labels)
{
  if (!path.parent_path().empty()) { fs::create_directories(path.parent_path()); }
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write allowed label list: " + path.string());
  }
  for (auto label : labels) {
    output << label << '\n';
  }
}

label_routing_plan make_label_routing_plan(const std::vector<uint32_t>& query_labels,
                                           const std::vector<int64_t>& data_label_counts,
                                           int64_t query_offset,
                                           int64_t query_count,
                                           int worker_count,
                                           double query_weight,
                                           double data_weight)
{
  std::unordered_map<uint32_t, std::vector<int64_t>> queries_by_label;
  for (int64_t i = 0; i < query_count; ++i) {
    auto query_id = query_offset + i;
    auto label = query_labels[static_cast<size_t>(query_id)];
    queries_by_label[label].push_back(query_id);
  }

  struct label_bucket {
    uint32_t label;
    double weight;
    int64_t query_count;
  };

  std::vector<label_bucket> buckets;
  buckets.reserve(queries_by_label.size());
  for (auto const& [label, query_ids] : queries_by_label) {
    auto label_size =
      (label != std::numeric_limits<uint32_t>::max() && label < data_label_counts.size())
        ? data_label_counts[label]
        : 0;
    auto weight =
      query_weight * static_cast<double>(query_ids.size()) +
      data_weight * static_cast<double>(std::max<int64_t>(1, label_size)) *
        static_cast<double>(query_ids.size());
    buckets.push_back(label_bucket{label, weight, static_cast<int64_t>(query_ids.size())});
  }

  std::sort(buckets.begin(), buckets.end(), [](const label_bucket& lhs, const label_bucket& rhs) {
    if (lhs.weight != rhs.weight) { return lhs.weight > rhs.weight; }
    return lhs.query_count > rhs.query_count;
  });

  label_routing_plan plan;
  plan.worker_query_ids.resize(static_cast<size_t>(worker_count));
  plan.worker_loads.assign(static_cast<size_t>(worker_count), 0.0);

  for (auto const& bucket : buckets) {
    auto worker_it = std::min_element(plan.worker_loads.begin(), plan.worker_loads.end());
    auto worker = static_cast<int>(std::distance(plan.worker_loads.begin(), worker_it));
    if (bucket.label != std::numeric_limits<uint32_t>::max()) {
      plan.label_to_worker[bucket.label] = worker;
    }
    auto const& query_ids = queries_by_label.at(bucket.label);
    auto& worker_queries = plan.worker_query_ids[static_cast<size_t>(worker)];
    worker_queries.insert(worker_queries.end(), query_ids.begin(), query_ids.end());
    *worker_it += bucket.weight;
  }

  return plan;
}

json read_json_file(const fs::path& path)
{
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Unable to open JSON file: " + path.string());
  }
  json value;
  input >> value;
  return value;
}

void write_json_file(const fs::path& path, const json& value)
{
  if (!path.parent_path().empty()) { fs::create_directories(path.parent_path()); }
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write JSON file: " + path.string());
  }
  output << value.dump(2);
}

pid_t spawn_worker(const fs::path& worker_executable, const worker_process& worker)
{
  auto pid = fork();
  if (pid < 0) {
    throw std::runtime_error("fork() failed: " + std::string(std::strerror(errno)));
  }
  if (pid == 0) {
    auto log_fd = ::open(worker.log_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (log_fd >= 0) {
      ::dup2(log_fd, STDOUT_FILENO);
      ::dup2(log_fd, STDERR_FILENO);
      ::close(log_fd);
    }

    auto visible_device = std::to_string(worker.device_id);
    ::setenv("CUDA_VISIBLE_DEVICES", visible_device.c_str(), 1);

    ::execl(worker_executable.c_str(),
            worker_executable.c_str(),
            "--config",
            worker.config_path.c_str(),
            static_cast<char*>(nullptr));
    std::perror("execl");
    _exit(127);
  }
  return pid;
}

void wait_for_worker_or_throw(const worker_process& worker)
{
  int status = 0;
  if (::waitpid(worker.pid, &status, 0) < 0) {
    throw std::runtime_error("waitpid() failed for worker log " + worker.log_path.string());
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("Worker failed for GPU " + std::to_string(worker.device_id) +
                             ". See log: " + worker.log_path.string());
  }
}

int main(int argc, char** argv)
{
  std::string config_file;
  if (argc < 3 || std::string(argv[1]) != "--config") {
    std::fprintf(stderr, "Usage: %s --config <config.json>\n", argv[0]);
    return 1;
  }
  config_file = argv[2];

  try {
    auto config = read_json_file(config_file);
    auto algorithms_to_run = config["algorithms_to_run"].get<std::vector<std::string>>();
    auto device_ids = config.value("device_ids", std::vector<int>{0});
    auto num_runs = config["num_runs"].get<int>();
    auto itopk_sizes = config["itopk_size"].get<std::vector<int>>();
    auto output_json_path = fs::path(config["output_json_file"].get<std::string>());
    auto label_aware_routing = config.value("label_aware_routing", false);
    auto label_routing_query_weight = config.value("label_routing_query_weight", 1.0);
    auto label_routing_data_weight = config.value("label_routing_data_weight", 1.0);

    if (device_ids.empty()) { throw std::runtime_error("device_ids cannot be empty"); }
    if (itopk_sizes.empty()) { throw std::runtime_error("itopk_size cannot be empty"); }

    for (const auto& algorithm : algorithms_to_run) {
      if (algorithm != "vecflow_mg" && algorithm != "vecflow_tagore_mg") {
        throw std::runtime_error(
          "VECFLOW_MG_BENCH only supports vecflow_mg and vecflow_tagore_mg. "
          "Use VECFLOW_BENCH for single-GPU algorithms.");
      }
    }

    auto query_path =
      config["data_dir"].get<std::string>() + config["query_fname"].get<std::string>();
    uint32_t total_query_rows = 0;
    uint32_t dim = 0;
    read_fbin_header(query_path, &total_query_rows, &dim);

    auto query_offset = config.value("query_offset", static_cast<int64_t>(0));
    auto query_count = config.value("query_count", static_cast<int64_t>(-1));
    if (query_offset < 0 || query_offset > static_cast<int64_t>(total_query_rows)) {
      throw std::runtime_error("query_offset is out of range");
    }
    auto effective_query_count = (query_count < 0)
                                   ? (static_cast<int64_t>(total_query_rows) - query_offset)
                                   : query_count;
    if (effective_query_count <= 0 ||
        query_offset + effective_query_count > static_cast<int64_t>(total_query_rows)) {
      throw std::runtime_error("query_count produces an invalid query range");
    }

    if (!output_json_path.parent_path().empty()) {
      fs::create_directories(output_json_path.parent_path());
    }

    auto worker_executable = get_self_executable_path().parent_path() / "VECFLOW_BENCH";
    if (!fs::exists(worker_executable)) {
      throw std::runtime_error("Unable to find VECFLOW_BENCH next to coordinator: " +
                               worker_executable.string());
    }

    auto run_id = std::to_string(
      std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch())
        .count());
    auto worker_root =
      (output_json_path.parent_path().empty() ? fs::current_path() : output_json_path.parent_path()) /
      ("mg_workers_" + run_id);
    fs::create_directories(worker_root);

    std::cout << "Loaded config: " << config_file << "\n";
    std::cout << "Query file: " << query_path << " (rows=" << total_query_rows
              << ", dim=" << dim << ")\n";
    std::cout << "Effective query range: [" << query_offset << ", "
              << (query_offset + effective_query_count) << ")\n";
    std::cout << "Devices: [ ";
    for (auto device_id : device_ids) { std::cout << device_id << " "; }
    std::cout << "]\n";
    std::cout << "Label-aware routing: " << (label_aware_routing ? "true" : "false") << "\n";

    std::vector<query_chunk> chunks;
    label_routing_plan routing_plan;
    if (label_aware_routing) {
      auto data_label_path =
        config["data_dir"].get<std::string>() + config["data_label_fname"].get<std::string>();
      auto query_label_path =
        config["data_dir"].get<std::string>() + config["query_label_fname"].get<std::string>();
      auto query_labels = read_first_query_labels(query_label_path);
      auto data_label_counts = read_data_label_counts(data_label_path);
      routing_plan = make_label_routing_plan(query_labels,
                                             data_label_counts,
                                             query_offset,
                                             effective_query_count,
                                             static_cast<int>(device_ids.size()),
                                             label_routing_query_weight,
                                             label_routing_data_weight);
      for (size_t worker = 0; worker < routing_plan.worker_query_ids.size(); ++worker) {
        std::cout << "  worker " << worker << " load=" << routing_plan.worker_loads[worker]
                  << ", queries=" << routing_plan.worker_query_ids[worker].size() << "\n";
      }
    } else {
      chunks = split_query_chunks(effective_query_count, static_cast<int>(device_ids.size()));
    }
    json results_json = json::array();

    for (const auto& algorithm_name : algorithms_to_run) {
      auto worker_algorithm = [&]() {
        if (algorithm_name == "vecflow_mg") { return std::string("vecflow"); }
        return std::string("vecflow_tagore");
      }();
      auto algorithm_dir = worker_root / sanitize_component(algorithm_name);
      fs::create_directories(algorithm_dir);
      auto run_config = config;
      double prebuild_seconds = 0.0;

      std::vector<worker_process> workers;
      workers.reserve(device_ids.size());

      std::cout << "\n=== " << algorithm_name << " ===\n";

      if (!label_aware_routing && should_prebuild_shared_cache(config, worker_algorithm)) {
        worker_process prebuild_worker;
        prebuild_worker.device_id = device_ids.front();
        prebuild_worker.query_offset = query_offset;
        prebuild_worker.query_count = std::min<int64_t>(effective_query_count, 1);
        prebuild_worker.config_path = algorithm_dir / "shared_prebuild.json";
        prebuild_worker.output_path = algorithm_dir / "shared_prebuild_results.json";
        prebuild_worker.log_path = algorithm_dir / "shared_prebuild.log";

        auto prebuild_config = config;
        prebuild_config["algorithms_to_run"] = json::array({worker_algorithm});
        prebuild_config["device_ids"] = json::array({0});
        prebuild_config["query_offset"] = prebuild_worker.query_offset;
        prebuild_config["query_count"] = prebuild_worker.query_count;
        prebuild_config["query_id_list_file"] = "";
        prebuild_config["num_runs"] = 1;
        prebuild_config["warmup_runs"] = 0;
        prebuild_config["ground_truth_fname"] = append_suffix_to_filename(
          config["ground_truth_fname"].get<std::string>(),
          "_" + sanitize_component(algorithm_name) + "_shared_prebuild");
        prebuild_config["output_json_file"] = prebuild_worker.output_path.string();
        write_json_file(prebuild_worker.config_path, prebuild_config);

        std::cout << "Prebuilding shared cache on GPU " << prebuild_worker.device_id
                  << " using " << primary_graph_cache_filename(config, worker_algorithm) << "\n";
        auto prebuild_start = std::chrono::steady_clock::now();
        prebuild_worker.pid = spawn_worker(worker_executable, prebuild_worker);
        wait_for_worker_or_throw(prebuild_worker);
        auto prebuild_end = std::chrono::steady_clock::now();
        prebuild_seconds =
          std::chrono::duration<double>(prebuild_end - prebuild_start).count();
        std::cout << "Shared cache prebuild finished in " << prebuild_seconds << " s\n";
      }
      run_config["force_rebuild"] = false;

      for (size_t i = 0; i < device_ids.size(); ++i) {
        worker_process worker;
        worker.device_id = device_ids[i];
        if (label_aware_routing) {
          worker.query_ids = routing_plan.worker_query_ids[i];
          worker.query_count = static_cast<int64_t>(worker.query_ids.size());
          if (worker.query_count == 0) { continue; }
          worker.query_offset = 0;
          for (auto const& [label, owner] : routing_plan.label_to_worker) {
            if (owner == static_cast<int>(i)) { worker.labels.push_back(label); }
          }
          std::sort(worker.labels.begin(), worker.labels.end());
        } else {
          auto chunk = chunks[i];
          if (chunk.size == 0) { continue; }
          worker.query_offset = query_offset + chunk.offset;
          worker.query_count = chunk.size;
        }

        auto cache_suffix =
          "_" + sanitize_component(algorithm_name) + "_d" + std::to_string(worker.device_id);
        auto query_suffix =
          label_aware_routing
            ? (cache_suffix + "_labelq_" + std::to_string(worker.query_count))
            : (cache_suffix + "_q" + std::to_string(worker.query_offset) + "_" +
               std::to_string(worker.query_count));

        auto worker_config = run_config;
        worker_config["algorithms_to_run"] = json::array({worker_algorithm});
        worker_config["device_ids"] = json::array({0});
        worker_config["query_offset"] = label_aware_routing ? 0 : worker.query_offset;
        worker_config["query_count"] = label_aware_routing ? -1 : worker.query_count;
        worker_config["ivf_graph_fname"] = choose_worker_cache_filename(
          run_config,
          "ivf_graph_fname",
          run_config["ivf_graph_fname"].get<std::string>(),
          cache_suffix,
          !label_aware_routing);
        worker_config["ivf_graph_tagore_fname"] = choose_worker_cache_filename(
          run_config,
          "ivf_graph_tagore_fname",
          std::string("ivf_graph_tagore.bin"),
          cache_suffix,
          !label_aware_routing);
        worker_config["ivf_bfs_fname"] = choose_worker_cache_filename(
          run_config,
          "ivf_bfs_fname",
          run_config["ivf_bfs_fname"].get<std::string>(),
          cache_suffix,
          !label_aware_routing);
        worker_config["cagra_index_fname"] = choose_worker_cache_filename(
          run_config,
          "cagra_index_fname",
          run_config["cagra_index_fname"].get<std::string>(),
          cache_suffix,
          !label_aware_routing);
        worker_config["ground_truth_fname"] = append_suffix_to_filename(
          config["ground_truth_fname"].get<std::string>(), query_suffix);

        worker.config_path = algorithm_dir / ("worker" + query_suffix + ".json");
        worker.output_path = algorithm_dir / ("worker" + query_suffix + "_results.json");
        worker.log_path = algorithm_dir / ("worker" + query_suffix + ".log");
        if (label_aware_routing) {
          worker.query_id_list_path = algorithm_dir / ("worker" + query_suffix + "_query_ids.txt");
          write_query_id_list(worker.query_id_list_path, worker.query_ids);
          worker_config["query_id_list_file"] = worker.query_id_list_path.string();
          worker.allowed_labels_path =
            algorithm_dir / ("worker" + query_suffix + "_allowed_labels.txt");
          write_allowed_labels(worker.allowed_labels_path, worker.labels);
          worker_config["allowed_labels_file"] = worker.allowed_labels_path.string();
        } else {
          worker_config["query_id_list_file"] = "";
          worker_config["allowed_labels_file"] = "";
        }
        worker_config["output_json_file"] = worker.output_path.string();
        write_json_file(worker.config_path, worker_config);

        if (label_aware_routing) {
          std::cout << "Launching worker on GPU " << worker.device_id << " for "
                    << worker.query_count << " label-routed queries across "
                    << worker.labels.size() << " labels\n";
        } else {
          std::cout << "Launching worker on GPU " << worker.device_id << " for queries ["
                    << worker.query_offset << ", " << (worker.query_offset + worker.query_count)
                    << ")\n";
        }
        worker.pid = spawn_worker(worker_executable, worker);
        workers.push_back(worker);
      }

      if (workers.empty()) {
        throw std::runtime_error("No worker processes were launched");
      }

      for (const auto& worker : workers) {
        wait_for_worker_or_throw(worker);
      }

      std::map<int, result_aggregate> aggregates;
      for (const auto& worker : workers) {
        auto worker_results = read_json_file(worker.output_path);
        for (const auto& row : worker_results) {
          auto itopk = row["itopk"].get<int>();
          auto qps = row["qps"].get<double>();
          auto recall = row["recall"].get<double>();
          auto build_seconds = row.value("build_seconds", 0.0);
          auto num_queries = row.value("num_queries", worker.query_count);
          if (qps <= 0.0) {
            throw std::runtime_error("Worker reported non-positive qps in " +
                                     worker.output_path.string());
          }
          auto search_seconds = (static_cast<double>(num_runs) * static_cast<double>(num_queries)) / qps;
          auto& aggregate = aggregates[itopk];
          aggregate.weighted_recall_sum += recall * static_cast<double>(num_queries);
          aggregate.max_build_seconds = std::max(aggregate.max_build_seconds, build_seconds);
          aggregate.max_search_seconds = std::max(aggregate.max_search_seconds, search_seconds);
          aggregate.num_queries += num_queries;
        }
      }

      for (const auto& [itopk, aggregate] : aggregates) {
        if (aggregate.num_queries != effective_query_count) {
          throw std::runtime_error("Aggregated query count mismatch for itopk=" +
                                   std::to_string(itopk));
        }
        auto recall =
          aggregate.weighted_recall_sum / static_cast<double>(aggregate.num_queries);
        auto qps =
          (static_cast<double>(num_runs) * static_cast<double>(aggregate.num_queries)) /
          aggregate.max_search_seconds;
        auto total_build_seconds = prebuild_seconds + aggregate.max_build_seconds;
        std::cout << "  - itopk=" << itopk << ", QPS=" << qps << ", Recall="
                  << recall << ", Build=" << total_build_seconds << " s\n";
        results_json.push_back({{"algorithm", algorithm_name},
                                {"itopk", itopk},
                                {"qps", qps},
                                {"recall", recall},
                                {"build_seconds", total_build_seconds},
                                {"num_devices", static_cast<int>(workers.size())},
                                {"num_queries", aggregate.num_queries},
                                {"query_offset", query_offset},
                                {"label_aware_routing", label_aware_routing}});
      }
    }

    write_json_file(output_json_path, results_json);
    std::cout << "\nWrote aggregated results to " << output_json_path << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
