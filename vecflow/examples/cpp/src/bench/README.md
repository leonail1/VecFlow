## Overview

The benchmark suite includes:

Standard search benchmark (`vecflow_bench.cu`) - Compare VecFlow against alternative filtering methods for large batch queries
Multi-GPU coordinator (`vecflow_mg_bench.cpp`) - Launch one single-GPU worker process per GPU and aggregate results
Multi-GPU C++ API smoke (`vecflow_mg_api_smoke.cu`) - Validate the in-library `build_multi_gpu/search_multi_gpu` path on multiple GPUs
Phoenix smoke test (`vecflow_phoenix_smoke.cpp`) - Minimal C++ validation for `libphoenix`

## Configuration

The benchmark tool uses JSON configuration files for flexibility:

### Standard Benchmark (`config.json`)

```json
{
  "data_dir": "/path/to/data/directory/",
  "data_fname": "base.fbin",
  "query_fname": "query.fbin",
  "data_label_fname": "base.txt",
  "query_label_fname": "query.txt",
  "algorithms_to_run": ["vecflow", "vecflow_tagore", "cagra_inline_filtering", "cagra_post_processing"],
  "itopk_size": [8, 16, 32, 64, 128, 256, 512],
  "spec_threshold": 2000,
  "graph_degree": 16,
  "topk": 10,
  "num_runs": 1000,
  "warmup_runs": 10,
  "ivf_graph_fname": "ivf_graph.bin",
  "ivf_graph_tagore_fname": "ivf_graph_tagore.bin",
  "ivf_bfs_fname": "ivf_bfs.bin",
  "cagra_index_fname": "cagra_16.bin",
  "ground_truth_fname": "ground_truth_10.bin",
  "output_json_file": "/path/to/output/results.json",
  "tagore_iterations": 10,
  "use_phoenix_graph_load": false,
  "use_phoenix_label_load": false,
  "phoenix_label_cache_bytes": 1073741824,
  "phoenix_label_dram_cache_bytes": 0,
  "phoenix_label_prefetch_max_bytes": 0,
  "phoenix_label_dataset_cache_bytes": 1073741824,
  "phoenix_label_dataset_dram_cache_bytes": 0,
  "phoenix_label_dataset_prefetch_max_bytes": 0,
  "phoenix_label_rebalance_interval_queries": 64,
  "query_offset": 0,
  "query_count": -1,
  "query_id_list_file": "",
  "label_aware_routing": false,
  "label_routing_query_weight": 1.0,
  "label_routing_data_weight": 1.0
}
```

### Multi-GPU Benchmark (`config_*_mg.json`)

Use the same JSON structure, but set `algorithms_to_run` to `vecflow_mg` / `vecflow_tagore_mg` and provide `device_ids`.

## Parameters

- `data_dir`: Directory containing dataset files
- `data_fname`: Filename for base vectors (typically `.fbin` or `.u8bin`)
- `query_fname`: Filename for query vectors (typically `.fbin` or `.u8bin`)
- `data_label_fname`: Filename for base vector labels (.txt or .spmat)
- `query_label_fname`: Filename for query vector labels (.txt or .spmat)
- `algorithms_to_run`: List of algorithms to benchmark. `vecflow_tagore` means Tagore-built per-label graphs plus cuVS `cagra::filtered_search` on the compatible graph layout.
- `itopk_size`: Array of internal topk sizes to test
- `spec_threshold`: Specificity threshold for VecFlow indexing
- `graph_degree`: Graph degree for CAGRA and VecFlow indices
- `topk`: Number of nearest neighbors to return
- `num_runs`: Number of benchmark runs to perform
- `warmup_runs`: Number of warmup runs before timing
- `ivf_graph_fname`: Filename for the cached VecFlow IVF graph
- `ivf_graph_tagore_fname`: Filename for the cached Tagore-built CAGRA-compatible IVF graph used by `vecflow_tagore`
- `ivf_bfs_fname`: Filename for IVF-BFS index
- `cagra_index_fname`: Filename for CAGRA index
- `ground_truth_fname`: Filename for ground truth data
- `output_json_file`: Path for benchmark results output
- `tagore_iterations`: Iterations passed to the Tagore GNN-Descent pipeline
- `use_phoenix_graph_load`: Optional. When `true`, cached VecFlow graph files are read directly into GPU memory through the Phoenix C++ interface instead of host I/O + `cudaMemcpy`
- `use_phoenix_label_load`: Optional. When `true`, cached VecFlow graph files stay on SSD and the single-GPU worker loads only the label-local graph rows needed by the current batch of queries through Phoenix. This takes precedence over `use_phoenix_graph_load`.
- `phoenix_label_cache_bytes`: Optional. HBM budget for the label-local Phoenix graph cache. `0` disables caching; the default benchmark value is `1073741824` (1 GiB).
- `phoenix_label_dram_cache_bytes`: Optional. Pinned-host DRAM budget for evicted or prefetched label-local graphs. `0` disables the DRAM tier.
- `phoenix_label_prefetch_max_bytes`: Optional. Maximum label-graph byte size eligible for one-label-ahead asynchronous prefetch from pinned-host DRAM or SSD/Phoenix into HBM. `0` disables prefetch.
- `phoenix_label_dataset_cache_bytes`: Optional. HBM budget for the packed label-local dataset cache used by Phoenix label-load search.
- `phoenix_label_dataset_dram_cache_bytes`: Optional. Pinned-host DRAM budget for evicted or prefetched packed label-local datasets.
- `phoenix_label_dataset_prefetch_max_bytes`: Optional. Maximum label-dataset byte size eligible for asynchronous prefetch. If unset, it falls back to the graph-prefetch budget.
- `phoenix_label_rebalance_interval_queries`: Optional. Period for Phoenix tier rebalancing based on `access_count / bytes`.
- `query_offset`: Optional absolute query start offset for worker-style slicing
- `query_count`: Optional query count for worker-style slicing (`-1` means use the rest)
- `query_id_list_file`: Optional text file with one absolute query id per line. When set, `VECFLOW_BENCH` runs exactly this non-contiguous query subset.
- `allowed_labels_file`: Optional text file with one label id per line (or comma-separated labels per line). When set, `VECFLOW_BENCH` filters both base/query labels to this owned-label subset before building the index.
- `label_aware_routing`: Optional. For `VECFLOW_MG_BENCH`, route queries to worker GPUs by label ownership instead of contiguous query slices.
- `label_routing_query_weight`: Optional. Query-count term in the weighted label-to-GPU bin packing policy.
- `label_routing_data_weight`: Optional. Base-label-size term in the weighted label-to-GPU bin packing policy.

## Binaries

- `VECFLOW_BENCH`: Single-GPU benchmark runner. It supports both contiguous query slicing through `query_offset` / `query_count` and arbitrary non-contiguous subsets through `query_id_list_file`.
- `VECFLOW_MG_BENCH`: Multi-process multi-GPU coordinator. Use this for `vecflow_mg` and `vecflow_tagore_mg`.
- `VECFLOW_MG_API_SMOKE`: In-process multi-GPU C++ API validation for `cuvs::neighbors::vecflow::build_multi_gpu` and `search_multi_gpu`.
- `VECFLOW_PHOENIX_SMOKE`: Minimal C++ Phoenix validation target.

### C++ Storage Stats

The in-library C++ API now exposes `cuvs::neighbors::vecflow::storage_stats(...)` for both
single-GPU `index<T>` and `multi_gpu_index<T>`. It reports:

- logical per-tier placement (`hbm`, `dram`, `ssd`) for graph and dataset labels
- resident cache occupancy (`resident_hbm`, `resident_dram`)
- multi-GPU worker ownership, worker device ids, and weighted worker loads

Example:

```cpp
auto index = cuvs::neighbors::vecflow::build(...);
auto stats = cuvs::neighbors::vecflow::storage_stats(index);
std::cout << "graph HBM labels=" << stats.graph.hbm.labels
          << ", graph SSD bytes=" << stats.graph.ssd.bytes << "\n";

auto mg_index = cuvs::neighbors::vecflow::build_multi_gpu(...);
auto mg_stats = cuvs::neighbors::vecflow::storage_stats(mg_index);
std::cout << "worker0 labels=" << mg_stats.worker_owned_labels[0]
          << ", worker0 load=" << mg_stats.worker_loads[0] << "\n";
```

## Usage

### Building the Benchmark Tools

First, make sure you've built the VecFlow library as described in the main examples README. Then compile the benchmark tools:

bash

```bash
# From the cpp directory
mkdir build
cd build
cmake ..
make
```

### Running Benchmarks

#### Standard Benchmark

bash

```bash
# Run with default config file
./VECFLOW_BENCH

# Or specify a custom config file
./VECFLOW_BENCH --config path/to/config.json

# Multi-GPU coordinator
./VECFLOW_MG_BENCH --config path/to/config_mg.json

# Multi-GPU C++ API smoke
./VECFLOW_MG_API_SMOKE

# Phoenix smoke test
./VECFLOW_PHOENIX_SMOKE [file_path] [device_id]
```

When `use_phoenix_graph_load` is enabled, the single-GPU worker process prints
`Loading matrix from ... through Phoenix` while loading a cached graph. The same field also works
through `VECFLOW_MG_BENCH`, because the coordinator forwards it into each worker config.

When `use_phoenix_label_load` is enabled, the worker keeps the cached graph on SSD and prints
`Loading rows [begin, end) ... through Phoenix` for each label-local graph or packed-dataset slice
loaded during search. If `phoenix_label_cache_bytes > 0`, the worker also keeps recently used
label-local graphs in HBM and prints cache hit / insert messages. If
`phoenix_label_dram_cache_bytes > 0`, it also keeps pinned-host DRAM graph copies for faster
re-promotion after HBM eviction. If `phoenix_label_dataset_cache_bytes > 0`, the worker applies
the same HBM/DRAM tiering to the packed label-local datasets, so `vecflow_tagore` and Phoenix
label-load no longer require the full graph-side dataset attachment to stay resident on GPU. At
build/load time, the worker now also performs an initial size-based placement that fills HBM first,
then pinned-host DRAM, and leaves the remaining labels on SSD. If
`phoenix_label_prefetch_max_bytes > 0`, it asynchronously prefetches one future non-HBM label
graph from pinned-host DRAM or SSD/Phoenix while the current label is being searched. If
`phoenix_label_dataset_prefetch_max_bytes > 0`, it does the same for packed label-local datasets.
If
`phoenix_label_rebalance_interval_queries > 0`, it periodically rebalances HBM/DRAM tiers using
an `access_count / bytes` score and prints score-based eviction / promotion messages for both graph
and dataset caches.
`VECFLOW_MG_BENCH` forwards the Phoenix-related fields into each worker config as well.

When `VECFLOW_MG_BENCH` is given shared cache filenames that do not exist yet, it now launches one
short single-GPU prebuild worker first, writes the shared cache once, and only then starts the
query-sharded workers. The reported multi-GPU `build_seconds` includes both this shared prebuild
time and the subsequent per-worker load time.

When `label_aware_routing` is enabled, `VECFLOW_MG_BENCH` reads the base/query label files,
computes a weighted label-to-GPU assignment, emits one `query_id_list_file` per worker, and then
launches each worker on its label-owned non-contiguous query subset. It also emits one
`allowed_labels_file` per worker, so each worker now builds only its owned labels instead of
rebuilding the full label space on every GPU.

This benchmark compares VecFlow against alternative filtering methods for large batch queries:

- VecFlow: Our optimized dual-structure approach
- VecFlow + Tagore: Tagore-built per-label graph with the default cuVS filtered-search path
- CAGRA with inline filtering: Using bitmap filters during search
- CAGRA with post-processing: Standard search then filter results
- Multi-GPU VecFlow: One worker process per GPU with either contiguous query sharding or label-aware routing in the coordinator

## Output

The benchmark outputs JSON files with detailed performance metrics:

- QPS (Queries Per Second)
- Recall@K

Example output:

json

```json
[
  {
    "algorithm": "vecflow",
    "itopk": 32,
    "qps": 24687.5,
    "recall": 0.9815
  },
  {
    "algorithm": "cagra_inline_filtering",
    "itopk": 32,
    "qps": 14325.6,
    "recall": 0.9756
  }
]
```

## Data Format Requirements

See the main examples README for details on data format requirements, including:

- Vector file formats (`.fbin`, `.u8bin`)
- Label formats (.txt and .spmat)
- Converting between label formats



## Cached Files

The benchmark generates cached files to speed up repeated runs:

- Index files for each algorithm
- Ground truth files for evaluation

These are automatically created if not present, or loaded if they exist.

## Notes

- `VECFLOW_BENCH` intentionally rejects `vecflow_mg` / `vecflow_tagore_mg`; use `VECFLOW_MG_BENCH` for those algorithms.
- `VECFLOW_MG_BENCH` can now generate per-worker `query_id_list_file` inputs automatically when `label_aware_routing=true`, so worker query subsets no longer need to be contiguous.
- `VECFLOW_MG_API_SMOKE` is the quickest way to verify that the in-library multi-GPU API is usable before running the heavier benchmark coordinator.
- `vecflow_tagore` now stores Tagore-built CAGRA-compatible graph rows with on-disk width `graph_degree`.
- Tagore native search has been removed from this benchmark suite; only the CAGRA-compatible Tagore path is supported.
- The Phoenix smoke target is useful for verifying build/load/invocation of the C++ interface, but actual Phoenix I/O correctness still depends on the host kernel / NVIDIA driver combination.
