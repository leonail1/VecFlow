# VecFlow / VecFlow-develop / FilteredVamana Repro Bundle (2026-04-18)

## Purpose
This folder preserves commands, configs, logs and small result artifacts needed to reproduce and verify the latest YFCC-10M experiments.

## Dataset
- Base: `/home/lzg/VecFlow-develop/vecflow/datasets/yfcc10M/base.10M.u8bin`
- Query: `/home/lzg/VecFlow-develop/vecflow/datasets/yfcc10M/query.public.single_label.u8bin` (61626 queries)
- Labels:
  - base: `base.metadata.10M.spmat`
  - query: `query.metadata.public.single_label.spmat`
- Ground truth: `GT.public.single_label.ibin`

## Saved Results
- `results/develop_bench_rebuild.json`: final VecFlow-develop BENCH result (QPS/recall/latency)
- `results/original_vecflow_results.json`: original VecFlow result
- `results/develop_timeseries.json`: develop timeseries diagnostics
- `results/yfcc10m_filteredvamana_reuse_index_repro.csv`: FilteredVamana recall/QPS sweep
- `results/*.png` and `results/*.csv`: generated plots and summaries

## Exact Commands Used
See `logs/*.log`, each starts with a `CMD:` line.
Representative commands:

```bash
# Original VecFlow benchmark
numactl --cpunodebind=1 --membind=1 /home/lzg/VecFlow/vecflow/examples/cpp/build-yfcc/VECFLOW_BENCH \
  --config /home/lzg/VecFlow-develop/experiment_results/three_way_comparison/configs/original_vecflow.json

# VecFlow-develop benchmark (rebuild)
numactl --cpunodebind=1 --membind=1 /home/lzg/VecFlow-develop/vecflow/examples/cpp/build/VECFLOW_BENCH \
  --config /home/lzg/VecFlow-develop/experiment_results/repro_bundle_20260418/configs/develop_bench_rebuild.json

# DiskANN filtered search
numactl --cpunodebind=1 --membind=1 /home/lzg/VecFlow-develop/thirdparty/DiskANN/build/apps/search_memory_index \
  --data_type uint8 --dist_fn l2 \
  --index_path_prefix /home/lzg/VecFlow-develop/experiment_results/three_way_comparison/diskann_index/yfcc10m \
  --query_file /home/lzg/VecFlow-develop/vecflow/datasets/yfcc10M/query.public.single_label.u8bin \
  --query_filters_file /home/lzg/VecFlow-develop/vecflow/datasets/yfcc10M/query_filters_diskann.txt \
  --gt_file /home/lzg/VecFlow-develop/experiment_results/three_way_comparison/diskann_gt/yfcc10m_gt.bin \
  --label_type uint -K 10 -T 52 -L 100
```

## Quick Run Check
Run the following command to verify this bundle is executable without re-running expensive indexing:

```bash
bash scripts/quick_check.sh
```

It validates JSON/CSV readability and regenerates a QPS comparison figure from saved timeseries JSON.
