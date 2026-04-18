#!/usr/bin/env python3
"""Run the three-way YFCC-10M experiment and plot QPS time-series comparison.

Systems:
  1) Original VecFlow (~/VecFlow)           - VECFLOW_BENCH aggregate QPS
  2) FilteredVamana (DiskANN, 52 threads)   - search_memory_index aggregate QPS
  3) VecFlow-develop (~/VecFlow-develop)    - VECFLOW_QPS_TIMESERIES per-second QPS

Output: Per-second QPS plot with VecFlow-develop time-series + horizontal
        baselines for Original VecFlow and FilteredVamana.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/lzg/VecFlow-develop')
ORIGINAL_ROOT = Path(os.path.expanduser('~/VecFlow'))
DISKANN_BUILD = ROOT / 'thirdparty/DiskANN/build'
DATA_DIR = ROOT / 'vecflow/datasets/yfcc10M'
OUTPUT_DIR = ROOT / 'experiment_results/three_way_comparison'


def run_cmd(cmd: list[str], label: str, log_dir: Path, timeout: int = 14400) -> subprocess.CompletedProcess:
    """Run a command with real-time streaming output to both console and log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{label.replace(" ", "_").lower()}.log'
    print(f'\n{"="*72}', flush=True)
    print(f'[{label}] {" ".join(str(c) for c in cmd[:8])} ...', flush=True)
    print(f'[{label}] Log: {log_path}', flush=True)
    t0 = time.time()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    with log_path.open('w') as log_f:
        log_f.write(f'CMD: {" ".join(str(c) for c in cmd)}\n')
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        try:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                sys.stdout.write(f'  [{label}] {line}')
                sys.stdout.flush()
                stdout_lines.append(line)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        elapsed = time.time() - t0
        log_f.write(f'\nRC: {proc.returncode}\nElapsed: {elapsed:.1f}s\n')
    print(f'[{label}] rc={proc.returncode} elapsed={elapsed:.1f}s', flush=True)
    stdout_text = ''.join(stdout_lines)
    if proc.returncode != 0:
        tail = stdout_text[-2000:]
        print(f'[{label}] ERROR:\n{tail}', flush=True)
    # Return a CompletedProcess-like object for compatibility
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout=stdout_text, stderr='')


# ---- Original VecFlow ----

def run_original_vecflow(log_dir: Path) -> float | None:
    bench_bin = ORIGINAL_ROOT / 'vecflow/examples/cpp/build-yfcc/VECFLOW_BENCH'
    if not bench_bin.exists():
        print(f'[Original VecFlow] Binary not found: {bench_bin}')
        return None

    config = {
        'data_dir': str(ORIGINAL_ROOT / 'vecflow/datasets/yfcc10M') + '/',
        'data_fname': 'base.10M.u8bin',
        'query_fname': 'query.public.single_label.u8bin',
        'data_label_fname': 'base.metadata.10M.spmat',
        'query_label_fname': 'query.metadata.public.single_label.spmat',
        'algorithms_to_run': ['vecflow'],
        'itopk_size': [64],
        'spec_threshold': 2000,
        'graph_degree': 16,
        'topk': 10,
        'num_runs': 5,
        'warmup_runs': 2,
        'ivf_graph_fname': 'ivf_graph_yfcc10m_t2000.bin',
        'ivf_bfs_fname': 'ivf_bfs_yfcc10m_t2000.bin',
        'cagra_index_fname': 'cagra_16.bin',
        'ground_truth_fname': 'GT.public.single_label.ibin',
        'output_json_file': str(OUTPUT_DIR / 'original_vecflow_results.json'),
        'force_rebuild': False,
    }
    config_path = OUTPUT_DIR / 'configs/original_vecflow.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + '\n')

    cmd = ['numactl', '--cpunodebind=1', '--membind=1',
           str(bench_bin), '--config', str(config_path)]
    proc = run_cmd(cmd, 'Original VecFlow', log_dir)
    if proc.returncode != 0:
        return None

    # Parse QPS from the JSON output
    result_path = Path(config['output_json_file'])
    if result_path.exists():
        results = json.loads(result_path.read_text())
        if isinstance(results, list):
            for entry in results:
                if entry.get('algorithm') == 'vecflow' and entry.get('itopk') == 64:
                    return float(entry.get('qps', 0))
    # Fallback: parse stdout
    for line in proc.stdout.splitlines():
        m = re.search(r'qps[=:]\s*([\d.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


# ---- FilteredVamana (DiskANN) ----

def run_diskann(log_dir: Path, threads: int = 52, search_l: int = 100) -> float | None:
    build_bin = DISKANN_BUILD / 'apps/build_memory_index'
    search_bin = DISKANN_BUILD / 'apps/search_memory_index'
    gt_bin = DISKANN_BUILD / 'apps/utils/compute_groundtruth_for_filters'

    for b in [build_bin, search_bin, gt_bin]:
        if not b.exists():
            print(f'[DiskANN] Binary not found: {b}')
            return None

    index_dir = OUTPUT_DIR / 'diskann_index'
    index_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(index_dir / 'yfcc10m')
    gt_file = str(OUTPUT_DIR / 'diskann_gt/yfcc10m_gt.bin')
    Path(gt_file).parent.mkdir(parents=True, exist_ok=True)
    result_prefix = str(OUTPUT_DIR / 'diskann_results/yfcc10m')
    (OUTPUT_DIR / 'diskann_results').mkdir(parents=True, exist_ok=True)

    numa = ['numactl', '--cpunodebind=1', '--membind=1']

    # Build index — DiskANN outputs the graph file at <prefix> (no extra suffix)
    index_file = prefix
    if not Path(index_file).exists():
        cmd = numa + [
            str(build_bin),
            '--data_type', 'uint8',
            '--dist_fn', 'l2',
            '--data_path', str(DATA_DIR / 'base.10M.u8bin'),
            '--index_path_prefix', prefix,
            '-R', '64',
            '-L', '100',
            '--FilteredLbuild', '100',
            '--alpha', '1.2',
            '-T', str(threads),
            '--label_file', str(DATA_DIR / 'base_labels_diskann.txt'),
            '--label_type', 'uint',
        ]
        proc = run_cmd(cmd, 'DiskANN Build', log_dir, timeout=14400)
        if proc.returncode != 0:
            return None
    else:
        print('[DiskANN Build] Index exists, skipping.')

    # Compute ground truth (use all threads to accelerate brute-force)
    if not Path(gt_file).exists():
        gt_env_cmd = ['env', f'OMP_NUM_THREADS={threads}'] + numa + [
            str(gt_bin),
            '--data_type', 'uint8',
            '--dist_fn', 'l2',
            '--base_file', str(DATA_DIR / 'base.10M.u8bin'),
            '--query_file', str(DATA_DIR / 'query.public.single_label.u8bin'),
            '--label_file', str(DATA_DIR / 'base_labels_diskann.txt'),
            '--filter_label_file', str(DATA_DIR / 'query_filters_diskann.txt'),
            '--gt_file', gt_file,
            '--K', '10',
        ]
        proc = run_cmd(gt_env_cmd, 'DiskANN GroundTruth', log_dir, timeout=14400)
        if proc.returncode != 0:
            return None
    else:
        print('[DiskANN GT] Ground truth exists, skipping.')

    # Search
    cmd = numa + [
        str(search_bin),
        '--data_type', 'uint8',
        '--dist_fn', 'l2',
        '--index_path_prefix', prefix,
        '--query_file', str(DATA_DIR / 'query.public.single_label.u8bin'),
        '--query_filters_file', str(DATA_DIR / 'query_filters_diskann.txt'),
        '--gt_file', gt_file,
        '--label_type', 'uint',
        '-K', '10',
        '-T', str(threads),
        '-L', str(search_l),
        '--result_path', result_prefix,
    ]
    proc = run_cmd(cmd, 'DiskANN Search', log_dir)
    if proc.returncode != 0:
        return None

    # Parse metrics row from DiskANN output table:
    #   Ls  QPS  Avg dist cmps  Mean Latency (mus)  99.9 Latency  Recall@10
    # Example:
    #   100    32394.98    286.78    1598.90    13135.42    25.75
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                l_val = int(parts[0])
                if l_val == search_l:
                    # Column 2 is QPS; column 6 is recall.
                    qps = float(parts[1])
                    return qps
            except (ValueError, IndexError):
                continue
    # Try parsing "QPS" from output
    for line in proc.stdout.splitlines():
        m = re.search(r'(?:QPS|Queries per second)[:\s]+([\d.]+)', line, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


# ---- VecFlow-develop (QPS time series) ----

def run_develop_timeseries(log_dir: Path) -> tuple[float | None, Path | None]:
    bench_bin = ROOT / 'vecflow/examples/cpp/build/VECFLOW_QPS_TIMESERIES'
    if not bench_bin.exists():
        print(f'[VecFlow-develop] Binary not found: {bench_bin}')
        return None, None

    ts_json = OUTPUT_DIR / 'develop_timeseries.json'
    config = {
        'data_dir': str(DATA_DIR) + '/',
        'data_fname': 'base.10M.u8bin',
        'query_fname': 'query.public.single_label.u8bin',
        'data_label_fname': 'base.metadata.10M.spmat',
        'query_label_fname': 'query.metadata.public.single_label.spmat',
        'algorithms_to_run': ['vecflow'],
        'itopk_size': [64],
        'spec_threshold': 2000,
        'graph_degree': 16,
        'topk': 10,
        'num_runs': 5,
        'warmup_runs': 2,
        'ivf_graph_fname': 'ivf_graph_yfcc10m_t2000.bin',
        'ivf_bfs_fname': 'ivf_bfs_yfcc10m_t2000.bin',
        'cagra_index_fname': 'cagra_16.bin',
        'ground_truth_fname': 'GT.public.single_label.ibin',
        'output_json_file': str(ts_json),
        'force_rebuild': False,
        'use_phoenix_label_load': True,
        'phoenix_label_cache_bytes': 16 * 1024**3,
        'phoenix_label_dram_cache_bytes': 16 * 1024**3,
        'phoenix_label_prefetch_max_bytes': 0,
        'phoenix_label_dataset_cache_bytes': 16 * 1024**3,
        'phoenix_label_dataset_dram_cache_bytes': 16 * 1024**3,
        'phoenix_label_dataset_prefetch_max_bytes': 0,
        'phoenix_label_rebalance_interval_queries': 64,
        'enable_bfs_tiered_cache': True,
        'bfs_hbm_cache_bytes': 16 * 1024**3,
        'bfs_dram_cache_bytes': 16 * 1024**3,
        'bfs_prefetch_max_bytes': 0,
        'bfs_rebalance_interval_queries': 64,
        'cascade_eviction': True,
        'query_label_mode': 'single',
        'telemetry_query_chunk_size': 16384,
    }
    config_path = OUTPUT_DIR / 'configs/develop_timeseries.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + '\n')

    cmd = ['numactl', '--cpunodebind=1', '--membind=1',
           str(bench_bin), '--config', str(config_path)]
    proc = run_cmd(cmd, 'VecFlow-develop Timeseries', log_dir)
    if proc.returncode != 0:
        return None, None

    if ts_json.exists():
        data = json.loads(ts_json.read_text())
        samples = data.get('samples', [])
        if samples:
            total_queries = float(samples[-1].get('queries_completed', 0))
            total_time = float(samples[-1].get('elapsed_seconds', 1))
            avg_qps = total_queries / total_time if total_time > 0 else 0
            return avg_qps, ts_json
    return None, None


def run_develop_bench(log_dir: Path) -> float | None:
    """Run VECFLOW_BENCH for aggregate QPS."""
    bench_bin = ROOT / 'vecflow/examples/cpp/build/VECFLOW_BENCH'
    if not bench_bin.exists():
        print(f'[VecFlow-develop BENCH] Binary not found: {bench_bin}')
        return None

    config = {
        'data_dir': str(DATA_DIR) + '/',
        'data_fname': 'base.10M.u8bin',
        'query_fname': 'query.public.single_label.u8bin',
        'data_label_fname': 'base.metadata.10M.spmat',
        'query_label_fname': 'query.metadata.public.single_label.spmat',
        'algorithms_to_run': ['vecflow'],
        'itopk_size': [64],
        'spec_threshold': 2000,
        'graph_degree': 16,
        'topk': 10,
        'num_runs': 5,
        'warmup_runs': 2,
        'ivf_graph_fname': 'ivf_graph_yfcc10m_t2000.bin',
        'ivf_bfs_fname': 'ivf_bfs_yfcc10m_t2000.bin',
        'cagra_index_fname': 'cagra_16.bin',
        'ground_truth_fname': 'GT.public.single_label.ibin',
        'output_json_file': str(OUTPUT_DIR / 'develop_bench_results.json'),
        'force_rebuild': False,
        'use_phoenix_label_load': True,
        'phoenix_label_cache_bytes': 16 * 1024**3,
        'phoenix_label_dram_cache_bytes': 16 * 1024**3,
        'phoenix_label_prefetch_max_bytes': 0,
        'phoenix_label_dataset_cache_bytes': 16 * 1024**3,
        'phoenix_label_dataset_dram_cache_bytes': 16 * 1024**3,
        'phoenix_label_dataset_prefetch_max_bytes': 0,
        'phoenix_label_rebalance_interval_queries': 64,
        'enable_bfs_tiered_cache': True,
        'bfs_hbm_cache_bytes': 16 * 1024**3,
        'bfs_dram_cache_bytes': 16 * 1024**3,
        'bfs_prefetch_max_bytes': 0,
        'bfs_rebalance_interval_queries': 64,
        'cascade_eviction': True,
    }
    config_path = OUTPUT_DIR / 'configs/develop_bench.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + '\n')

    cmd = ['numactl', '--cpunodebind=1', '--membind=1',
           str(bench_bin), '--config', str(config_path)]
    proc = run_cmd(cmd, 'VecFlow-develop BENCH', log_dir)
    if proc.returncode != 0:
        return None

    result_path = Path(config['output_json_file'])
    if result_path.exists():
        results = json.loads(result_path.read_text())
        if isinstance(results, list):
            for entry in results:
                if entry.get('algorithm') == 'vecflow' and entry.get('itopk') == 64:
                    return float(entry.get('qps', 0))
    return None


# ---- Plotting ----

def plot_three_way(ts_json_path: Path | None,
                   original_qps: float | None,
                   diskann_qps: float | None,
                   develop_qps: float | None,
                   output_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # If we have timeseries data, plot per-second QPS curve
    has_ts = False
    ts_x, ts_y = [], []
    if ts_json_path and ts_json_path.exists():
        data = json.loads(ts_json_path.read_text())
        samples = data.get('samples', [])
        if len(samples) >= 2:
            has_ts = True
            total_sec = float(samples[-1]['elapsed_seconds'])
            bucket_sec = 1.0
            n_buckets = int(math.floor(total_sec / bucket_sec))
            times = [float(s['elapsed_seconds']) for s in samples]
            queries = [float(s['queries_completed']) for s in samples]
            import bisect
            for b in range(n_buckets):
                t0 = b * bucket_sec
                t1 = (b + 1) * bucket_sec
                # Interpolate queries_completed at t0 and t1
                def interp_q(t):
                    if t <= times[0]: return queries[0]
                    if t >= times[-1]: return queries[-1]
                    idx = bisect.bisect_right(times, t)
                    lt, rt = times[idx-1], times[idx]
                    lq, rq = queries[idx-1], queries[idx]
                    if rt <= lt: return rq
                    alpha = (t - lt) / (rt - lt)
                    return lq + (rq - lq) * alpha
                q0 = interp_q(t0)
                q1 = interp_q(t1)
                ts_x.append(t1)
                ts_y.append((q1 - q0) / bucket_sec)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    if has_ts:
        ax.plot(ts_x, ts_y, color='#2196F3', linewidth=1.2, label='VecFlow-develop (per-second)', alpha=0.85)

    max_x = max(ts_x) if ts_x else 60
    if original_qps is not None:
        ax.axhline(y=original_qps, color='#FF9800', linestyle='--', linewidth=2,
                    label=f'Original VecFlow: {original_qps:,.0f} QPS')
    if diskann_qps is not None:
        ax.axhline(y=diskann_qps, color='#F44336', linestyle='-.', linewidth=2,
                    label=f'FilteredVamana (52T): {diskann_qps:,.0f} QPS')
    if develop_qps is not None:
        ax.axhline(y=develop_qps, color='#4CAF50', linestyle=':', linewidth=2,
                    label=f'VecFlow-develop avg: {develop_qps:,.0f} QPS')

    ax.set_xlabel('Elapsed Time (seconds)', fontsize=12)
    ax.set_ylabel('Queries Per Second (QPS)', fontsize=12)
    ax.set_title('YFCC-10M (61626 queries, 192d, uint8, top-10) — Three-Way QPS Comparison', fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    png_path = output_dir / 'three_way_qps_comparison.png'
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f'\nPlot saved to {png_path}')

    # Also save a summary CSV
    csv_path = output_dir / 'three_way_summary.csv'
    with csv_path.open('w') as f:
        f.write('system,qps\n')
        if original_qps is not None:
            f.write(f'Original VecFlow,{original_qps:.2f}\n')
        if diskann_qps is not None:
            f.write(f'FilteredVamana (52T),{diskann_qps:.2f}\n')
        if develop_qps is not None:
            f.write(f'VecFlow-develop,{develop_qps:.2f}\n')
    print(f'Summary CSV saved to {csv_path}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-original', action='store_true')
    parser.add_argument('--skip-diskann', action='store_true')
    parser.add_argument('--skip-develop', action='store_true')
    parser.add_argument('--skip-develop-timeseries', action='store_true')
    parser.add_argument('--diskann-search-l', type=int, default=100)
    parser.add_argument('--threads', type=int, default=52)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT_DIR / 'logs'

    original_qps = None
    diskann_qps = None
    develop_qps = None
    ts_json_path = None

    # Step 1: Original VecFlow
    if not args.skip_original:
        print('\n' + '='*72)
        print('STEP 1: Original VecFlow')
        print('='*72)
        original_qps = run_original_vecflow(log_dir)
        print(f'  => Original VecFlow QPS: {original_qps}')
    else:
        # Try loading from existing results
        rp = OUTPUT_DIR / 'original_vecflow_results.json'
        if rp.exists():
            results = json.loads(rp.read_text())
            if isinstance(results, list):
                for entry in results:
                    if entry.get('algorithm') == 'vecflow' and entry.get('itopk') == 64:
                        original_qps = float(entry.get('qps', 0))

    # Step 2: FilteredVamana
    if not args.skip_diskann:
        print('\n' + '='*72)
        print('STEP 2: FilteredVamana (DiskANN)')
        print('='*72)
        diskann_qps = run_diskann(log_dir, threads=args.threads, search_l=args.diskann_search_l)
        print(f'  => FilteredVamana QPS: {diskann_qps}')

    # Step 3: VecFlow-develop aggregate
    if not args.skip_develop:
        print('\n' + '='*72)
        print('STEP 3: VecFlow-develop (BENCH aggregate)')
        print('='*72)
        develop_qps = run_develop_bench(log_dir)
        print(f'  => VecFlow-develop QPS: {develop_qps}')

    # Step 4: VecFlow-develop time series
    if not args.skip_develop_timeseries:
        print('\n' + '='*72)
        print('STEP 4: VecFlow-develop (QPS time-series)')
        print('='*72)
        ts_qps, ts_json_path = run_develop_timeseries(log_dir)
        if develop_qps is None:
            develop_qps = ts_qps
        print(f'  => VecFlow-develop time-series avg QPS: {ts_qps}')

    # Step 5: Plot
    print('\n' + '='*72)
    print('STEP 5: Plotting')
    print('='*72)
    plot_three_way(ts_json_path, original_qps, diskann_qps, develop_qps, OUTPUT_DIR)

    # Summary
    summary = {
        'original_vecflow_qps': original_qps,
        'filtered_vamana_qps': diskann_qps,
        'vecflow_develop_qps': develop_qps,
        'dataset': 'YFCC-10M',
        'queries': 61626,
        'dims': 192,
        'dtype': 'uint8',
        'topk': 10,
    }
    summary_path = OUTPUT_DIR / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'\n{"="*72}')
    print('Final Summary:')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
