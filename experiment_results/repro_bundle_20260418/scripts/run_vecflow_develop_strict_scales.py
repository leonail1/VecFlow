#!/usr/bin/env python3
"""VecFlow-develop 多规模 QPS 基准测试批量运行脚本。

针对 YFCC-10M 数据集，在不同查询规模（60k / 600k / 6M）和缓存消融模式
（full / no_hbm / no_dram / no_hbm_dram）下批量运行 VECFLOW_BENCH，
收集 QPS、Recall、延迟等指标并汇总为 CSV。

典型用法::

    python3 run_vecflow_develop_strict_scales.py \\
        --modes full no_dram --scales 60000 600000 6000000

输出目录结构::

    results/<output-subdir>/
        configs/   # 每次运行的 JSON 配置
        raw/       # VECFLOW_BENCH 原始 JSON 结果
        plots/     # 可选的 QPS 时序图
        logs/      # 运行日志（含 stdout/stderr）
        summary.csv
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/lzg/VecFlow-develop')
BENCH_BIN = ROOT / 'vecflow/examples/cpp/build/VECFLOW_BENCH'
DATA_DIR = ROOT / 'vecflow/datasets/yfcc10M'
BUNDLE_ROOT = ROOT / 'experiment_results/repro_bundle_20260418'

DEFAULT_SCALES = [60000, 600000, 6000000]
DEFAULT_BASE_QUERIES = 61626


def run_cmd(cmd: list[str], log_path: Path, timeout: int = 86400) -> int:
    """执行外部命令，实时写入日志并同步输出到 stdout。

    Args:
        cmd: 要执行的命令及参数列表。
        log_path: 日志文件路径，父目录会自动创建。
        timeout: 超时秒数，超时后强制终止子进程。

    Returns:
        子进程的退出码。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as log_f:
        log_f.write(f'CMD: {" ".join(str(c) for c in cmd)}\n')
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        start = time.time()
        try:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                sys.stdout.write(line)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        elapsed = time.time() - start
        log_f.write(f'\nRC: {proc.returncode}\nElapsed: {elapsed:.1f}s\n')
    return proc.returncode


def cache_profile(mode: str) -> dict[str, int | bool]:
    """根据缓存消融模式生成 Phoenix / BFS 分层缓存配置。

    Args:
        mode: 缓存模式，可选值：
            - ``'full'``: HBM 16 GiB + DRAM 16 GiB（完整缓存）。
            - ``'no_hbm'``: 禁用 HBM，仅 DRAM 16 GiB。
            - ``'no_dram'``: 禁用 DRAM，仅 HBM 16 GiB。
            - ``'no_hbm_dram'``: 同时禁用 HBM 和 DRAM（仅 SSD）。

    Returns:
        可直接合并到 VECFLOW_BENCH JSON 配置的字典。

    Raises:
        ValueError: 不支持的 mode 值。
    """
    gib = 1024 ** 3
    if mode == 'full':
        hbm = 16 * gib
        dram = 16 * gib
    elif mode == 'no_hbm':
        hbm = 0
        dram = 16 * gib
    elif mode == 'no_dram':
        hbm = 16 * gib
        dram = 0
    elif mode == 'no_hbm_dram':
        hbm = 0
        dram = 0
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    return {
        'use_phoenix_label_load': True,
        'phoenix_label_cache_bytes': hbm,
        'phoenix_label_dram_cache_bytes': dram,
        'phoenix_label_prefetch_max_bytes': 0,
        'phoenix_label_dataset_cache_bytes': hbm,
        'phoenix_label_dataset_dram_cache_bytes': dram,
        'phoenix_label_dataset_prefetch_max_bytes': 0,
        'phoenix_label_rebalance_interval_queries': 64,
        'enable_bfs_tiered_cache': True,
        'bfs_hbm_cache_bytes': hbm,
        'bfs_dram_cache_bytes': dram,
        'bfs_prefetch_max_bytes': 0,
        'bfs_rebalance_interval_queries': 64,
        'cascade_eviction': True,
    }


def build_config(scale_queries: int, mode: str, output_json_file: Path, num_runs: int) -> dict:
    """构建 VECFLOW_BENCH 的完整 JSON 配置字典。

    Args:
        scale_queries: 目标查询总量（如 60000、600000、6000000）。
        mode: 缓存消融模式，传递给 ``cache_profile()``。
        output_json_file: VECFLOW_BENCH 写入结果的 JSON 路径。
        num_runs: 搜索循环执行次数（= ceil(scale_queries / base_queries)）。

    Returns:
        可序列化为 JSON 并传给 ``VECFLOW_BENCH --config`` 的配置字典。
    """
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
        'num_runs': num_runs,
        'warmup_runs': 2,
        'min_num_runs': num_runs,
        'max_num_runs': num_runs,
        'stability_window': num_runs,
        'stable_subset_size': num_runs,
        'qps_stability_rel_tol': 0.0,
        'latency_stability_rel_tol': 0.0,
        'trend_guard_rel_tol': 0.0,
        'ivf_graph_fname': 'ivf_graph_yfcc10m_t2000.bin',
        'ivf_bfs_fname': 'ivf_bfs_yfcc10m_t2000.bin',
        'cagra_index_fname': 'cagra_16.bin',
        'ground_truth_fname': 'GT.public.single_label.ibin',
        'output_json_file': str(output_json_file),
        'force_rebuild': False,
        'skip_recall': False,
        'query_count': -1,
        'query_offset': 0,
        'target_query_scale': scale_queries,
    }
    config.update(cache_profile(mode))
    return config


def parse_result(result_path: Path) -> dict:
    """解析 VECFLOW_BENCH 输出的 JSON 结果文件。

    优先匹配 ``algorithm='vecflow'`` 且 ``itopk=64`` 的条目；
    若未找到则返回第一个条目。

    Args:
        result_path: VECFLOW_BENCH 写入的 JSON 结果文件路径。

    Returns:
        匹配的结果字典，包含 qps、recall、search_seconds 等字段。

    Raises:
        RuntimeError: JSON 格式不符合预期（非列表或为空）。
    """
    payload = json.loads(result_path.read_text())
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f'Unexpected bench output format: {result_path}')
    for entry in payload:
        if entry.get('algorithm') == 'vecflow' and int(entry.get('itopk', -1)) == 64:
            return entry
    return payload[0]


def main() -> int:
    """主入口：解析参数 → 遍历 mode×scale 组合 → 运行基准测试 → 汇总 CSV。

    流程:
        1. 对每个 ``(mode, scale)`` 组合计算所需 ``num_runs``。
        2. 生成配置 JSON 并调用 ``numactl`` + ``VECFLOW_BENCH``。
        3. 成功则解析结果并可选调用绘图脚本；失败则记录错误。
        4. 所有组合跑完后将结果合并写入 ``summary.csv``
           （增量合并，不覆盖已有行）。

    Returns:
        0 表示全部成功，2 表示二进制不存在，3 表示无有效结果。
    """
    parser = argparse.ArgumentParser(description='Run VecFlow-develop strict 1s QPS time-series for YFCC scales')
    parser.add_argument('--scales', nargs='+', type=int, default=DEFAULT_SCALES)
    parser.add_argument('--modes', nargs='+', choices=['full', 'no_hbm', 'no_dram', 'no_hbm_dram'], default=['full'])
    parser.add_argument('--base-queries', type=int, default=DEFAULT_BASE_QUERIES)
    parser.add_argument('--output-subdir', default='yfcc_strict_timeseries_20260418')
    parser.add_argument('--plot-script', default=str(BUNDLE_ROOT / 'scripts/plot_vecflow_qps_timeseries.py'))
    args = parser.parse_args()

    if not BENCH_BIN.exists():
        print(f'Binary not found: {BENCH_BIN}', file=sys.stderr)
        return 2

    run_root = BUNDLE_ROOT / 'results' / args.output_subdir
    config_dir = run_root / 'configs'
    raw_dir = run_root / 'raw'
    plot_dir = run_root / 'plots'
    log_dir = run_root / 'logs'
    for path in (config_dir, raw_dir, plot_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    def extract_failure(log_path: Path) -> str:
        if not log_path.exists():
            return 'log_missing'
        text = log_path.read_text(errors='ignore')
        rc = 'unknown'
        for line in text.splitlines():
            if line.startswith('RC: '):
                rc = line.split(':', 1)[1].strip()
        if 'cudaErrorIllegalAddress' in text:
            return f'RC={rc}; cudaErrorIllegalAddress'
        return f'RC={rc}; failed'

    def make_row(mode: str,
                 scale: int,
                 num_runs: int,
                 result_path: Path,
                 config_path: Path,
                 log_path: Path,
                 status: str,
                 error: str = '',
                 entry: dict | None = None) -> dict:
        if entry is None:
            return {
                'mode': mode,
                'scale_queries': scale,
                'num_runs': num_runs,
                'status': status,
                'error': error,
                'qps': '',
                'recall': '',
                'search_seconds': '',
                'strict_samples': '',
                'result_json': str(result_path),
                'config_json': str(config_path),
                'log_file': str(log_path),
            }
        strict_samples = entry.get('strict_qps_time_series', [])
        return {
            'mode': mode,
            'scale_queries': scale,
            'num_runs': num_runs,
            'status': status,
            'error': error,
            'qps': float(entry.get('qps', 0.0)),
            'recall': float(entry.get('recall', 0.0)) if entry.get('recall') is not None else float('nan'),
            'search_seconds': float(entry.get('search_seconds', 0.0)),
            'result_json': str(result_path),
            'config_json': str(config_path),
            'log_file': str(log_path),
        }

    for mode in args.modes:
        for scale in args.scales:
            num_runs = max(1, int(math.ceil(float(scale) / float(args.base_queries))))
            tag = f'{mode}_q{scale}_runs{num_runs}'
            config_path = config_dir / f'{tag}.json'
            result_path = raw_dir / f'{tag}.json'
            log_path = log_dir / f'{tag}.log'

            config = build_config(scale, mode, result_path, num_runs)
            config_path.write_text(json.dumps(config, indent=2) + '\n')

            cmd = [
                'numactl', '--cpunodebind=1', '--membind=1',
                str(BENCH_BIN), '--config', str(config_path)
            ]
            print(f'\n=== Running {tag} ===')
            rc = run_cmd(cmd, log_path)
            if rc != 0:
                print(f'Run failed for {tag}, rc={rc}', file=sys.stderr)
                summary_rows.append(make_row(mode,
                                             scale,
                                             num_runs,
                                             result_path,
                                             config_path,
                                             log_path,
                                             status='failed',
                                             error=extract_failure(log_path)))
                continue

            entry = parse_result(result_path)
            summary_rows.append(make_row(mode,
                                         scale,
                                         num_runs,
                                         result_path,
                                         config_path,
                                         log_path,
                                         status='ok',
                                         entry=entry))

            plot_cmd = [
                sys.executable,
                args.plot_script,
                '--series', f'{tag}={result_path}',
                '--algorithm', 'vecflow',
                '--itopk', '64',
                '--output-dir', str(plot_dir / tag)
            ]
            plot_rc = run_cmd(plot_cmd, log_dir / f'{tag}_plot.log', timeout=1800)
            if plot_rc != 0:
                print(f'Plot failed for {tag}, rc={plot_rc}', file=sys.stderr)

    summary_path = run_root / 'summary.csv'
    if summary_rows:
        merged: dict[tuple[str, str, str], dict] = {}
        if summary_path.exists():
            with summary_path.open('r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (str(row.get('mode', '')),
                           str(row.get('scale_queries', '')),
                           str(row.get('num_runs', '')))
                    merged[key] = row
        for row in summary_rows:
            key = (str(row['mode']), str(row['scale_queries']), str(row['num_runs']))
            merged[key] = row

        merged_rows = [merged[key] for key in sorted(merged.keys(), key=lambda k: (k[0], int(k[1]), int(k[2])))]
        with summary_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
            writer.writeheader()
            writer.writerows(merged_rows)
        print(f'Wrote summary: {summary_path}')
    else:
        print('No successful runs were recorded.', file=sys.stderr)
        return 3

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
