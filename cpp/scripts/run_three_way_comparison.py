#!/usr/bin/env python3
"""Three-way QPS comparison: Original VecFlow vs FilteredVamana vs VecFlow-develop.

Dataset: YFCC-10M, ~61626 single-label queries, 192d uint8, topk=10.
"""
import argparse
import json
import os
import subprocess
import struct
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], label: str, log_path: Path | None = None, timeout: int = 3600) -> dict:
    print(f"\n{'='*60}")
    print(f"[{label}] Running: {' '.join(cmd[:6])} ...")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT after {timeout}s")
        return {"ok": False, "error": "timeout", "elapsed": timeout}
    elapsed = time.time() - start
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            f.write(f"=== STDOUT ===\n{result.stdout}\n=== STDERR ===\n{result.stderr}\n")
    print(f"[{label}] Done in {elapsed:.1f}s, rc={result.returncode}")
    if result.returncode != 0:
        tail = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
        print(f"[{label}] FAILED:\n{tail}")
    return {
        "ok": result.returncode == 0,
        "elapsed": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def parse_vecflow_qps(stdout: str, itopk: int = 64) -> list[dict]:
    """Extract QPS and recall from VECFLOW_BENCH stdout for given itopk."""
    results = []
    for line in stdout.splitlines():
        if "vecflow" in line.lower() and f"itopk={itopk}" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p.startswith("qps="):
                    qps = float(p.split("=")[1].rstrip(","))
                    results.append({"qps": qps})
    return results


def parse_vecflow_json_results(json_path: Path, algo: str = "vecflow", itopk: int = 64) -> dict | None:
    if not json_path.exists():
        return None
    with json_path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        for entry in data:
            if entry.get("algorithm") == algo and entry.get("itopk") == itopk:
                return entry
    return None


def parse_diskann_search_output(stdout: str) -> list[dict]:
    """Parse DiskANN search_memory_index output to extract QPS/recall per L."""
    results = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 7:
            try:
                L = int(parts[0])
                recall = float(parts[3])
                qps = float(parts[5]) if '.' in parts[5] else float(parts[5])
                latency = float(parts[6]) if len(parts) > 6 else 0
                results.append({"L": L, "recall": recall, "qps": qps, "latency_us": latency})
            except (ValueError, IndexError):
                continue
    return results


def run_original_vecflow(args, output_dir: Path) -> dict:
    """Run original VecFlow at ~/VecFlow."""
    bench_bin = Path(args.original_vecflow_dir) / "vecflow/examples/cpp/build-yfcc/VECFLOW_BENCH"
    data_dir = Path(args.original_vecflow_dir) / "vecflow/datasets/yfcc10M/"
    if not bench_bin.exists():
        return {"ok": False, "error": f"Binary not found: {bench_bin}"}

    config = {
        "data_dir": str(data_dir) + "/",
        "data_fname": "base.10M.u8bin",
        "query_fname": "query.public.single_label.u8bin",
        "data_label_fname": "base.metadata.10M.spmat",
        "query_label_fname": "query.metadata.public.single_label.spmat",
        "algorithms_to_run": ["vecflow"],
        "itopk_size": [64],
        "spec_threshold": 2000,
        "graph_degree": 16,
        "topk": 10,
        "num_runs": 5,
        "warmup_runs": 2,
        "ivf_graph_fname": "ivf_graph_yfcc10m_t2000.bin",
        "ivf_bfs_fname": "ivf_bfs_yfcc10m_t2000.bin",
        "cagra_index_fname": "cagra_16.bin",
        "ground_truth_fname": "GT.public.single_label.ibin",
        "output_json_file": str(output_dir / "original_vecflow_results.json"),
        "force_rebuild": False,
    }
    config_path = output_dir / "original_vecflow_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    cmd = [
        "numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}",
        str(bench_bin), "--config", str(config_path),
    ]
    result = run_cmd(cmd, "Original VecFlow", output_dir / "logs/original_vecflow.log")
    if result["ok"]:
        parsed = parse_vecflow_json_results(Path(config["output_json_file"]))
        result["parsed"] = parsed
    return result


def run_diskann_filtered(args, output_dir: Path) -> dict:
    """Run filtered DiskANN (FilteredVamana)."""
    diskann_build_dir = Path(args.diskann_build_dir)
    data_dir = Path(args.develop_dir) / "vecflow/datasets/yfcc10M"
    build_bin = diskann_build_dir / "apps/build_memory_index"
    search_bin = diskann_build_dir / "apps/search_memory_index"
    gt_bin = diskann_build_dir / "apps/utils/compute_groundtruth_for_filters"

    index_prefix = str(output_dir / "diskann_index/yfcc10m_filtered")
    gt_file = str(output_dir / "diskann_gt/yfcc10m_filtered_gt.bin")
    result_prefix = str(output_dir / "diskann_results/yfcc10m_filtered")

    Path(index_prefix).parent.mkdir(parents=True, exist_ok=True)
    Path(gt_file).parent.mkdir(parents=True, exist_ok=True)
    Path(result_prefix).parent.mkdir(parents=True, exist_ok=True)

    numa = ["numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}"]

    # Build index
    if not Path(index_prefix + "_filterVamana.index").exists() or args.force_rebuild:
        build_cmd = numa + [
            str(build_bin),
            "--data_type", "uint8",
            "--dist_fn", "l2",
            "--data_path", str(data_dir / "base.10M.u8bin"),
            "--index_path_prefix", index_prefix,
            "-R", "64",
            "-L", "100",
            "--FilteredLbuild", "100",
            "--alpha", "1.2",
            "-T", str(args.threads),
            "--label_file", str(data_dir / "base_labels_diskann.txt"),
            "--label_type", "uint",
        ]
        build_result = run_cmd(build_cmd, "DiskANN Build", output_dir / "logs/diskann_build.log", timeout=7200)
        if not build_result["ok"]:
            return build_result
    else:
        print("[DiskANN Build] Skipping - index already exists")

    # Compute ground truth
    if not Path(gt_file).exists() or args.force_rebuild:
        gt_cmd = numa + [
            str(gt_bin),
            "--data_type", "uint8",
            "--dist_fn", "l2",
            "--base_file", str(data_dir / "base.10M.u8bin"),
            "--query_file", str(data_dir / "query.public.single_label.u8bin"),
            "--label_file", str(data_dir / "base_labels_diskann.txt"),
            "--filter_label_file", str(data_dir / "query_filters_diskann.txt"),
            "--gt_file", gt_file,
            "--K", "10",
        ]
        gt_result = run_cmd(gt_cmd, "DiskANN GT", output_dir / "logs/diskann_gt.log", timeout=7200)
        if not gt_result["ok"]:
            return gt_result
    else:
        print("[DiskANN GT] Skipping - ground truth exists")

    # Search with multiple L values
    search_cmd = numa + [
        str(search_bin),
        "--data_type", "uint8",
        "--dist_fn", "l2",
        "--index_path_prefix", index_prefix,
        "--query_file", str(data_dir / "query.public.single_label.u8bin"),
        "--query_filters_file", str(data_dir / "query_filters_diskann.txt"),
        "--gt_file", gt_file,
        "--label_type", "uint",
        "-K", "10",
        "-T", str(args.threads),
        "-L", "10", "20", "40", "80", "100", "120", "160",
        "--result_path", result_prefix,
    ]
    search_result = run_cmd(search_cmd, "DiskANN Search", output_dir / "logs/diskann_search.log")
    if search_result["ok"]:
        parsed = parse_diskann_search_output(search_result["stdout"])
        search_result["parsed"] = parsed
    return search_result


def run_develop_vecflow(args, output_dir: Path) -> dict:
    """Run VecFlow-develop."""
    bench_bin = Path(args.develop_dir) / "vecflow/examples/cpp/build/VECFLOW_BENCH"
    data_dir = Path(args.develop_dir) / "vecflow/datasets/yfcc10M/"
    if not bench_bin.exists():
        return {"ok": False, "error": f"Binary not found: {bench_bin}"}

    config = {
        "data_dir": str(data_dir) + "/",
        "data_fname": "base.10M.u8bin",
        "query_fname": "query.public.single_label.u8bin",
        "data_label_fname": "base.metadata.10M.spmat",
        "query_label_fname": "query.metadata.public.single_label.spmat",
        "algorithms_to_run": ["vecflow"],
        "itopk_size": [64],
        "spec_threshold": 2000,
        "graph_degree": 16,
        "topk": 10,
        "num_runs": 5,
        "warmup_runs": 2,
        "ivf_graph_fname": "ivf_graph_yfcc10m_t2000.bin",
        "ivf_bfs_fname": "ivf_bfs_yfcc10m_t2000.bin",
        "cagra_index_fname": "cagra_16.bin",
        "ground_truth_fname": "GT.public.single_label.ibin",
        "output_json_file": str(output_dir / "develop_vecflow_results.json"),
        "force_rebuild": False,
        "use_phoenix_label_load": True,
        "phoenix_label_cache_bytes": 4294967296,
        "phoenix_label_dram_cache_bytes": 8589934592,
        "phoenix_label_dataset_cache_bytes": 4294967296,
        "phoenix_label_dataset_dram_cache_bytes": 8589934592,
        "phoenix_label_prefetch_max_bytes": 67108864,
        "phoenix_label_dataset_prefetch_max_bytes": 67108864,
        "phoenix_label_rebalance_interval_queries": 64,
        "bfs_hbm_cache_bytes": 2147483648,
        "bfs_dram_cache_bytes": 4294967296,
        "bfs_prefetch_max_bytes": 67108864,
        "bfs_rebalance_interval_queries": 64,
        "cascade_eviction": True,
        "enable_bfs_tiered_cache": True,
    }
    config_path = output_dir / "develop_vecflow_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    cmd = [
        "numactl", f"--cpunodebind={args.numa_node}", f"--membind={args.numa_node}",
        str(bench_bin), "--config", str(config_path),
    ]
    result = run_cmd(cmd, "VecFlow-develop", output_dir / "logs/develop_vecflow.log")
    if result["ok"]:
        parsed = parse_vecflow_json_results(Path(config["output_json_file"]))
        result["parsed"] = parsed
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Three-way QPS comparison")
    parser.add_argument("--original-vecflow-dir", default=os.path.expanduser("~/VecFlow"))
    parser.add_argument("--develop-dir", default=os.path.expanduser("~/VecFlow-develop"))
    parser.add_argument("--diskann-build-dir",
                        default=os.path.expanduser("~/VecFlow-develop/thirdparty/DiskANN/build"))
    parser.add_argument("--output-dir", default=os.path.expanduser("~/VecFlow-develop/experiment_results/three_way_comparison"))
    parser.add_argument("--numa-node", type=int, default=1)
    parser.add_argument("--threads", type=int, default=52)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-diskann", action="store_true")
    parser.add_argument("--skip-develop", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    if not args.skip_original:
        result = run_original_vecflow(args, output_dir)
        summary["original_vecflow"] = {
            "ok": result["ok"],
            "elapsed": result.get("elapsed", 0),
            "parsed": result.get("parsed"),
        }
    if not args.skip_diskann:
        result = run_diskann_filtered(args, output_dir)
        summary["filtered_vamana"] = {
            "ok": result["ok"],
            "elapsed": result.get("elapsed", 0),
            "parsed": result.get("parsed"),
        }
    if not args.skip_develop:
        result = run_develop_vecflow(args, output_dir)
        summary["vecflow_develop"] = {
            "ok": result["ok"],
            "elapsed": result.get("elapsed", 0),
            "parsed": result.get("parsed"),
        }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Summary written to {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
