#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize VecFlow benchmark result JSON files')
    parser.add_argument('--config-dir', required=True, help='Directory containing benchmark configs')
    parser.add_argument('--result-glob', required=True, help='Glob for result JSON files')
    parser.add_argument('--output-json', help='Optional JSON output path')
    return parser.parse_args()


def bytes_to_gib(value: int | float | None) -> float:
    if value is None:
        return 0.0
    return float(value) / (1024 ** 3)


def nested_get(obj: dict[str, Any], *keys: str, default: Any = 0) -> Any:
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def summarize_one(config_path: Path, result_path: Path) -> list[dict[str, Any]]:
    if not config_path.exists() or not result_path.exists():
        return []
    config = json.loads(config_path.read_text())
    results = load_result_rows(result_path)
    rows: list[dict[str, Any]] = []
    for result in results:
        external = result.get('external_metrics', {})
        storage = result.get('storage_stats', {})
        graph = storage.get('graph', {})
        dataset = storage.get('dataset', {})
        bfs = storage.get('bfs', {})
        latency = result.get('latency_ms', {})
        gpu_latency = result.get('gpu_latency_ms', {})
        cpu_overhead = result.get('cpu_overhead_ms', {})
        cache_counters = result.get('cache_counters', {})

        row = {
            'name': config_path.stem,
            'itopk': result.get('itopk', 0),
            'query_label_mode': result.get('query_label_mode', config.get('query_label_mode', 'single')),
            'ground_truth_label_mode': result.get('ground_truth_label_mode', 'any'),
            'skip_recall': result.get('skip_recall', False),
            'cascade_eviction': config.get('cascade_eviction', True),
            'use_phoenix_label_load': config.get('use_phoenix_label_load', False),
            'use_phoenix_graph_load': config.get('use_phoenix_graph_load', False),
            'bfs_hbm_budget_gib': bytes_to_gib(config.get('bfs_hbm_cache_bytes')),
            'bfs_dram_budget_gib': bytes_to_gib(config.get('bfs_dram_cache_bytes')),
            'phoenix_hbm_budget_gib': bytes_to_gib(config.get('phoenix_label_cache_bytes')),
            'phoenix_dram_budget_gib': bytes_to_gib(config.get('phoenix_label_dram_cache_bytes')),
            'query_count': config.get('query_count', -1),
            'num_queries': result.get('num_queries', 0),
            'num_runs': result.get('num_runs', config.get('num_runs', 0)),
            'warmup_runs': result.get('warmup_runs', config.get('warmup_runs', 0)),
            'qps': result.get('qps', 0.0),
            'recall': result.get('recall', None),
            'build_seconds': result.get('build_seconds', 0.0),
            'search_seconds': result.get('search_seconds', 0.0),
            'latency_avg_ms': latency.get('avg_ms', 0.0),
            'latency_p50_ms': latency.get('p50_ms', 0.0),
            'latency_p95_ms': latency.get('p95_ms', 0.0),
            'latency_max_ms': latency.get('max_ms', 0.0),
            'gpu_latency_avg_ms': gpu_latency.get('avg_ms', 0.0),
            'gpu_latency_p50_ms': gpu_latency.get('p50_ms', 0.0),
            'gpu_latency_p95_ms': gpu_latency.get('p95_ms', 0.0),
            'gpu_latency_max_ms': gpu_latency.get('max_ms', 0.0),
            'cpu_overhead_avg_ms': cpu_overhead.get('avg_ms', 0.0),
            'cpu_overhead_p50_ms': cpu_overhead.get('p50_ms', 0.0),
            'cpu_overhead_p95_ms': cpu_overhead.get('p95_ms', 0.0),
            'cpu_overhead_max_ms': cpu_overhead.get('max_ms', 0.0),
            'peak_rss_gib': bytes_to_gib(external.get('peak_rss_bytes')),
            'peak_gpu_memory_gib': float(external.get('peak_gpu_memory_mib', 0.0)) / 1024.0,
            'peak_rxpci_mb_s': float(external.get('peak_rxpci_mb_s', 0.0)),
            'peak_txpci_mb_s': float(external.get('peak_txpci_mb_s', 0.0)),
            'read_gib': bytes_to_gib(external.get('read_bytes')),
            'write_gib': bytes_to_gib(external.get('write_bytes')),
            'storage_access_events': storage.get('access_events', 0),
            'graph_hbm_gib': bytes_to_gib(nested_get(graph, 'hbm', 'bytes')),
            'graph_dram_gib': bytes_to_gib(nested_get(graph, 'dram', 'bytes')),
            'graph_ssd_gib': bytes_to_gib(nested_get(graph, 'ssd', 'bytes')),
            'dataset_hbm_gib': bytes_to_gib(nested_get(dataset, 'hbm', 'bytes')),
            'dataset_dram_gib': bytes_to_gib(nested_get(dataset, 'dram', 'bytes')),
            'dataset_ssd_gib': bytes_to_gib(nested_get(dataset, 'ssd', 'bytes')),
            'bfs_hbm_gib': bytes_to_gib(nested_get(bfs, 'hbm', 'bytes')),
            'bfs_dram_gib': bytes_to_gib(nested_get(bfs, 'dram', 'bytes')),
            'bfs_ssd_gib': bytes_to_gib(nested_get(bfs, 'ssd', 'bytes')),
            'graph_hbm_labels': nested_get(graph, 'hbm', 'labels'),
            'graph_dram_labels': nested_get(graph, 'dram', 'labels'),
            'graph_ssd_labels': nested_get(graph, 'ssd', 'labels'),
            'dataset_hbm_labels': nested_get(dataset, 'hbm', 'labels'),
            'dataset_dram_labels': nested_get(dataset, 'dram', 'labels'),
            'dataset_ssd_labels': nested_get(dataset, 'ssd', 'labels'),
            'bfs_hbm_labels': nested_get(bfs, 'hbm', 'labels'),
            'bfs_dram_labels': nested_get(bfs, 'dram', 'labels'),
            'bfs_ssd_labels': nested_get(bfs, 'ssd', 'labels'),
        }

        for tier in ['phoenix_graph', 'phoenix_dataset', 'tiered_bfs']:
            counters = cache_counters.get(tier, {})
            for key in ['access_events', 'hbm_hits', 'dram_hits', 'ssd_loads', 'hbm_evictions', 'dram_evictions']:
                row[f'{tier}_{key}'] = counters.get(key, 0)
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    config_dir = Path(args.config_dir)
    rows: list[dict[str, Any]] = []
    for result_name in sorted(glob.glob(args.result_glob)):
        result_path = Path(result_name)
        config_path = config_dir / result_path.name
        rows.extend(summarize_one(config_path, result_path))

    if not rows:
        raise SystemExit('No matching results found')

    rows.sort(key=lambda row: (row['name'], int(row.get('itopk', 0))))

    headers = [
        'name', 'itopk', 'query_label_mode', 'qps', 'recall', 'build_seconds', 'search_seconds',
        'latency_avg_ms', 'gpu_latency_avg_ms', 'cpu_overhead_avg_ms', 'peak_gpu_memory_gib',
        'graph_hbm_gib', 'dataset_hbm_gib', 'bfs_hbm_gib', 'bfs_dram_gib', 'bfs_ssd_gib',
        'phoenix_graph_hbm_hits', 'phoenix_dataset_hbm_hits', 'tiered_bfs_hbm_hits', 'tiered_bfs_ssd_loads'
    ]
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f'{value:.4f}')
            elif value is None:
                values.append('null')
            else:
                values.append(str(value))
        print('| ' + ' | '.join(values) + ' |')

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
