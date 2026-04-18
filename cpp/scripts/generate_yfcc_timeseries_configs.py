#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path('/home/lzg/VecFlow-develop')
OUT_DIR = ROOT / 'cpp/scripts/generated_timeseries_configs'
DATA_DIR = ROOT / 'vecflow/datasets/yfcc10M'

COMMON = {
    'data_dir': str(DATA_DIR) + '/',
    'data_fname': 'base.10M.u8bin',
    'data_label_fname': 'base.metadata.10M.spmat',
    'graph_degree': 16,
    'spec_threshold': 1000,
    'topk': 10,
    'itopk_size': [64],
    'num_runs': 10,
    'warmup_runs': 5,
    'tagore_iterations': 10,
    'force_rebuild': False,
    'query_label_mode': 'single',
    'use_phoenix_graph_load': False,
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
    'cascade_eviction': False,
    'algorithms_to_run': ['vecflow'],
    'ivf_graph_fname': 'qcount_probe_e3c4_ivf_graph.bin',
    'ivf_graph_tagore_fname': 'qcount_probe_e3c4_ivf_graph_tagore.bin',
    'ivf_bfs_fname': 'qcount_probe_e3c4_ivf_bfs.bin',
    'telemetry_query_chunk_size': 16384,
}

CONFIGS = {
    'yfcc_timeseries_q61626_it64': {
        'query_fname': 'query.public.single_label.u8bin',
        'query_label_fname': 'query.metadata.public.single_label.spmat',
        'query_count': 61626,
        'output_json_file': '/tmp/yfcc_timeseries_q61626_it64.json',
    },
    'yfcc_timeseries_q600k_it64': {
        'query_fname': 'query.public.single_label.random_600k.u8bin',
        'query_label_fname': 'query.metadata.public.single_label.random_600k.spmat',
        'query_count': 600000,
        'output_json_file': '/tmp/yfcc_timeseries_q600k_it64.json',
    },
    'yfcc_timeseries_q6000k_it64': {
        'query_fname': 'query.public.single_label.random_6000k.u8bin',
        'query_label_fname': 'query.metadata.public.single_label.random_6000k.spmat',
        'query_count': 6000000,
        'output_json_file': '/tmp/yfcc_timeseries_q6000k_it64.json',
    },
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, extra in CONFIGS.items():
        path = OUT_DIR / f'{name}.json'
        payload = dict(COMMON)
        payload.update(extra)
        path.write_text(json.dumps(payload, indent=2) + '\n')
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
