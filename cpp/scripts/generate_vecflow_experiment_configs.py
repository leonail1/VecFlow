#!/usr/bin/env python3
import json
import os
import struct
from pathlib import Path

ROOT = Path('/home/lzg/VecFlow-develop')
DATA_DIR = ROOT / 'vecflow/datasets/yfcc10M'
OUT_DIR = ROOT / 'cpp/scripts/generated_configs'
MULTI_QUERY_LABEL_FILE = DATA_DIR / 'query.metadata.public.100K.spmat'
YFCC_MULTI_QUERY_ID_LIST = ROOT / 'cpp/scripts/yfcc_multi_query_ids_full.txt'

YFCC_SINGLE_QUERY_COUNT = 61_626
YFCC_MULTI_QUERY_TARGET_COUNT = 61_626
WIKIANN_QUERY_COUNT = 10_000
NUM_RUNS = 10
WARMUP_RUNS = 5
MIN_NUM_RUNS = 3
STABILITY_WINDOW = 5
STABLE_SUBSET_SIZE = 3
QPS_STABILITY_REL_TOL = 0.10
LATENCY_STABILITY_REL_TOL = 0.10
TREND_GUARD_REL_TOL = 0.10
ITOPK_SIZES = [64, 128, 256, 512]
SHARED_ARTIFACT_PREFIX = 'validate_shared'

RESOURCE_CONFIGS = {
    'c1': {'hbm_gb': 8, 'dram_gb': 16},
    'c2': {'hbm_gb': 8, 'dram_gb': 32},
    'c3': {'hbm_gb': 8, 'dram_gb': 64},
    'c4': {'hbm_gb': 16, 'dram_gb': 16},
    'c5': {'hbm_gb': 16, 'dram_gb': 32},
    'c6': {'hbm_gb': 16, 'dram_gb': 64},
}


def gb(value: int) -> int:
    return value * 1024 * 1024 * 1024


def qtag(count: int) -> str:
    return f'q{count}'


def read_spmat_indptr(path: Path) -> tuple[int, tuple[int, ...]]:
    with path.open('rb') as f:
        nrow, _ncol, _nnz = struct.unpack('qqq', f.read(24))
        indptr = struct.unpack(f'{nrow + 1}q', f.read(8 * (nrow + 1)))
    return nrow, indptr


def ensure_multi_query_id_list(path: Path, query_label_path: Path, target_count: int) -> int:
    nrow, indptr = read_spmat_indptr(query_label_path)
    selected = [str(row) for row in range(nrow) if (indptr[row + 1] - indptr[row]) >= 2][:target_count]
    if not selected:
        raise RuntimeError(f'No multi-label queries with at least two labels were found in {query_label_path}')
    path.write_text('\n'.join(selected) + '\n')
    if len(selected) < target_count:
        print(f'Using all available multi-label YFCC queries: requested {target_count}, found {len(selected)}')
    return len(selected)


def shared_base(data_dir: Path) -> dict:
    return {
        'data_dir': str(data_dir) + '/',
        'algorithms_to_run': ['vecflow'],
        'itopk_size': ITOPK_SIZES,
        'spec_threshold': 1000,
        'graph_degree': 16,
        'topk': 10,
        'num_runs': NUM_RUNS,
        'warmup_runs': WARMUP_RUNS,
        'min_num_runs': MIN_NUM_RUNS,
        'max_num_runs': NUM_RUNS,
        'stability_window': STABILITY_WINDOW,
        'stable_subset_size': STABLE_SUBSET_SIZE,
        'qps_stability_rel_tol': QPS_STABILITY_REL_TOL,
        'latency_stability_rel_tol': LATENCY_STABILITY_REL_TOL,
        'trend_guard_rel_tol': TREND_GUARD_REL_TOL,
        'tagore_iterations': 10,
        'force_rebuild': False,
    }


def single_label_query_base() -> dict:
    return {
        'query_fname': 'query.public.single_label.u8bin',
        'query_label_fname': 'query.metadata.public.single_label.spmat',
        'ground_truth_fname': 'GT.public.single_label.ibin',
        'query_offset': 0,
        'query_count': YFCC_SINGLE_QUERY_COUNT,
        'query_label_mode': 'single',
    }


def multi_label_query_base(mode: str, query_id_list: Path) -> dict:
    return {
        'query_fname': 'query.public.100K.u8bin',
        'query_label_fname': 'query.metadata.public.100K.spmat',
        'ground_truth_fname': 'GT.public.ibin',
        'query_id_list_file': str(query_id_list),
        'query_offset': 0,
        'query_count': -1,
        'query_label_mode': mode,
    }


def no_tier_fields() -> dict:
    return {
        'use_phoenix_graph_load': False,
        'use_phoenix_label_load': False,
        'cascade_eviction': True,
        'phoenix_label_cache_bytes': gb(1),
        'phoenix_label_dram_cache_bytes': 0,
        'phoenix_label_dataset_cache_bytes': gb(1),
        'phoenix_label_dataset_dram_cache_bytes': 0,
    }


def tiered_fields(hbm_gb: int, dram_gb: int, cascade_eviction: bool) -> dict:
    return {
        'use_phoenix_graph_load': False,
        'use_phoenix_label_load': True,
        'cascade_eviction': cascade_eviction,
        'bfs_hbm_cache_bytes': gb(hbm_gb),
        'bfs_dram_cache_bytes': gb(dram_gb),
        'bfs_prefetch_max_bytes': 0,
        'bfs_rebalance_interval_queries': 64,
        'phoenix_label_cache_bytes': gb(hbm_gb),
        'phoenix_label_dram_cache_bytes': gb(dram_gb),
        'phoenix_label_prefetch_max_bytes': 0,
        'phoenix_label_dataset_cache_bytes': gb(hbm_gb),
        'phoenix_label_dataset_dram_cache_bytes': gb(dram_gb),
        'phoenix_label_dataset_prefetch_max_bytes': 0,
        'phoenix_label_rebalance_interval_queries': 64,
    }


def with_artifacts(config: dict, name: str) -> dict:
    config = dict(config)
    config['ivf_graph_fname'] = f'{SHARED_ARTIFACT_PREFIX}_ivf_graph.bin'
    config['ivf_graph_tagore_fname'] = f'{SHARED_ARTIFACT_PREFIX}_ivf_graph_tagore.bin'
    config['ivf_bfs_fname'] = f'{SHARED_ARTIFACT_PREFIX}_ivf_bfs.bin'
    config['cagra_index_fname'] = f'{SHARED_ARTIFACT_PREFIX}_cagra.bin'
    config['output_json_file'] = f'/tmp/{name}.json'
    return config


def write_config(name: str, config: dict) -> None:
    path = OUT_DIR / f'{name}.json'
    path.write_text(json.dumps(with_artifacts(config, name), indent=2) + '\n')


def find_first_match(directory: Path, patterns: list[str], exclude_substring: str | None = None) -> str | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        for match in matches:
            if not match.is_file():
                continue
            if exclude_substring and exclude_substring in match.name:
                continue
            return match.name
    return None


def discover_wikiann_dataset_dir() -> Path | None:
    candidates: list[Path] = []
    env_dir = os.environ.get('VECFLOW_WIKIANN_DATA_DIR')
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend([
        ROOT / 'vecflow/datasets/wikiann',
        ROOT / 'vecflow/datasets/WikiANN',
    ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def maybe_generate_wikiann_configs() -> None:
    dataset_dir = discover_wikiann_dataset_dir()
    if dataset_dir is None:
        print('Skipping WikiANN config generation: dataset directory not found')
        return

    files = {
        'data_fname': find_first_match(dataset_dir, ['base*.fbin', 'base*.u8bin', 'base*.ibin']),
        'data_label_fname': find_first_match(dataset_dir, ['base*.spmat']),
        'query_fname': find_first_match(dataset_dir, ['query*.fbin', 'query*.u8bin', 'query*.ibin']),
        'query_label_fname': find_first_match(dataset_dir, ['query*.spmat']),
        'single_ground_truth_fname': find_first_match(dataset_dir, ['GT*single*.ibin', 'gt*single*.ibin']),
        'multi_ground_truth_fname': find_first_match(dataset_dir, ['GT*.ibin', 'gt*.ibin'], exclude_substring='single'),
    }
    if any(value is None for value in files.values()):
        print(f'Skipping WikiANN config generation: incomplete dataset files under {dataset_dir}')
        return

    base = {
        **shared_base(dataset_dir),
        'data_fname': files['data_fname'],
        'data_label_fname': files['data_label_fname'],
    }
    single = {
        'query_fname': files['query_fname'],
        'query_label_fname': files['query_label_fname'],
        'ground_truth_fname': files['single_ground_truth_fname'],
        'query_offset': 0,
        'query_count': WIKIANN_QUERY_COUNT,
        'query_label_mode': 'single',
    }
    multi_and_greedy = {
        'query_fname': files['query_fname'],
        'query_label_fname': files['query_label_fname'],
        'ground_truth_fname': files['multi_ground_truth_fname'],
        'query_offset': 0,
        'query_count': WIKIANN_QUERY_COUNT,
        'query_label_mode': 'multi_and_greedy',
    }
    multi_and_parallel = dict(multi_and_greedy)
    multi_and_parallel['query_label_mode'] = 'multi_and_parallel'

    write_config('wikiann_e6_notier_q10000', {
        **base,
        **single,
        **no_tier_fields(),
    })
    for cfg_name, limits in RESOURCE_CONFIGS.items():
        write_config(f'wikiann_e7_{cfg_name}_direct_q10000', {
            **base,
            **single,
            **tiered_fields(limits['hbm_gb'], limits['dram_gb'], False),
        })
        write_config(f'wikiann_e8_{cfg_name}_cascade_q10000', {
            **base,
            **single,
            **tiered_fields(limits['hbm_gb'], limits['dram_gb'], True),
        })
    write_config('wikiann_e11_c5_multi_and_greedy_q10000', {
        **base,
        **multi_and_greedy,
        **tiered_fields(RESOURCE_CONFIGS['c5']['hbm_gb'], RESOURCE_CONFIGS['c5']['dram_gb'], True),
    })
    write_config('wikiann_e12_c5_multi_and_parallel_q10000', {
        **base,
        **multi_and_parallel,
        **tiered_fields(RESOURCE_CONFIGS['c5']['hbm_gb'], RESOURCE_CONFIGS['c5']['dram_gb'], True),
    })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_config in list(OUT_DIR.glob('yfcc_e*.json')) + list(OUT_DIR.glob('wikiann_e*.json')):
        stale_config.unlink()

    yfcc_multi_query_count = ensure_multi_query_id_list(
        YFCC_MULTI_QUERY_ID_LIST, MULTI_QUERY_LABEL_FILE, YFCC_MULTI_QUERY_TARGET_COUNT
    )
    yfcc_single_suffix = qtag(YFCC_SINGLE_QUERY_COUNT)
    yfcc_multi_suffix = qtag(yfcc_multi_query_count)

    write_config(f'yfcc_e2_notier_{yfcc_single_suffix}', {
        **shared_base(DATA_DIR),
        'data_fname': 'base.10M.u8bin',
        'data_label_fname': 'base.metadata.10M.spmat',
        **single_label_query_base(),
        **no_tier_fields(),
    })

    for cfg_name, limits in RESOURCE_CONFIGS.items():
        write_config(f'yfcc_e3_{cfg_name}_direct_{yfcc_single_suffix}', {
            **shared_base(DATA_DIR),
            'data_fname': 'base.10M.u8bin',
            'data_label_fname': 'base.metadata.10M.spmat',
            **single_label_query_base(),
            **tiered_fields(limits['hbm_gb'], limits['dram_gb'], False),
        })
        write_config(f'yfcc_e4_{cfg_name}_cascade_{yfcc_single_suffix}', {
            **shared_base(DATA_DIR),
            'data_fname': 'base.10M.u8bin',
            'data_label_fname': 'base.metadata.10M.spmat',
            **single_label_query_base(),
            **tiered_fields(limits['hbm_gb'], limits['dram_gb'], True),
        })

    write_config(f'yfcc_e9_c5_multi_or_{yfcc_multi_suffix}', {
        **shared_base(DATA_DIR),
        'data_fname': 'base.10M.u8bin',
        'data_label_fname': 'base.metadata.10M.spmat',
        **multi_label_query_base('multi_or', YFCC_MULTI_QUERY_ID_LIST),
        **tiered_fields(RESOURCE_CONFIGS['c5']['hbm_gb'], RESOURCE_CONFIGS['c5']['dram_gb'], True),
    })
    write_config(f'yfcc_e10_c5_multi_and_greedy_{yfcc_multi_suffix}', {
        **shared_base(DATA_DIR),
        'data_fname': 'base.10M.u8bin',
        'data_label_fname': 'base.metadata.10M.spmat',
        **multi_label_query_base('multi_and_greedy', YFCC_MULTI_QUERY_ID_LIST),
        **tiered_fields(RESOURCE_CONFIGS['c5']['hbm_gb'], RESOURCE_CONFIGS['c5']['dram_gb'], True),
    })
    write_config(f'yfcc_e10_c5_multi_and_parallel_{yfcc_multi_suffix}', {
        **shared_base(DATA_DIR),
        'data_fname': 'base.10M.u8bin',
        'data_label_fname': 'base.metadata.10M.spmat',
        **multi_label_query_base('multi_and_parallel', YFCC_MULTI_QUERY_ID_LIST),
        **tiered_fields(RESOURCE_CONFIGS['c5']['hbm_gb'], RESOURCE_CONFIGS['c5']['dram_gb'], True),
    })

    maybe_generate_wikiann_configs()


if __name__ == '__main__':
    main()
