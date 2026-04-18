#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path('/home/lzg/VecFlow-develop')
RUNNER = ROOT / 'cpp/scripts/run_vecflow_experiment.py'
BINARY = ROOT / 'vecflow/examples/cpp/build/VECFLOW_BENCH'
CONFIG_DIR = ROOT / 'cpp/scripts/generated_configs'
LOG_DIR = Path('/tmp/vecflow_wikiann1m_suite_logs')


def experiment_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r'_e(\d+)_', path.stem)
    experiment_id = int(match.group(1)) if match else 1_000_000
    return experiment_id, path.stem


def discover_configs() -> list[Path]:
    configs = [path for path in CONFIG_DIR.glob('wikiann1m_e*.json') if path.is_file()]
    return sorted(configs, key=experiment_sort_key)


def append_suffix_to_filename(filename: str, suffix: str) -> str:
    path = Path(filename)
    return str(path.with_name(f'{path.stem}{suffix}{path.suffix}'))


def collect_artifacts(config: dict[str, Any]) -> list[Path]:
    data_dir = Path(config['data_dir'])
    artifacts: set[Path] = set()
    for key in ['ivf_graph_fname', 'ivf_graph_tagore_fname', 'ivf_bfs_fname', 'cagra_index_fname']:
        value = config.get(key)
        if value:
            artifacts.add(data_dir / value)
    graph_fname = config.get('ivf_graph_fname', '')
    bfs_fname = config.get('ivf_bfs_fname', '')
    if graph_fname:
        artifacts.add(data_dir / append_suffix_to_filename(graph_fname, '_dataset'))
    if bfs_fname:
        artifacts.add(data_dir / append_suffix_to_filename(bfs_fname, '_dataset'))
    return sorted(artifacts)


def cleanup_for_config(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    for artifact in collect_artifacts(config):
        if artifact.exists():
            artifact.unlink()


def result_path_for_config(config_path: Path) -> Path:
    config = json.loads(config_path.read_text())
    return Path(config['output_json_file'])


def is_completed_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    if isinstance(payload, list):
        return bool(payload) and all(isinstance(item, dict) for item in payload)
    return isinstance(payload, dict)


def run_config(config_path: Path) -> None:
    log_path = LOG_DIR / f'{config_path.stem}.log'
    cmd = [
        sys.executable,
        str(RUNNER),
        '--binary', str(BINARY),
        '--config', str(config_path),
        '--numa-node', '1',
        '--poll-seconds', '1.0',
        '--cleanup-indexes',
        '--log-file', str(log_path),
    ]
    print(f'>>> Running {config_path.name}')
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    configs = discover_configs()
    if not configs:
        raise SystemExit(f'No WikiANN-1M configs found in {CONFIG_DIR}')

    for config_path in configs:
        result_path = result_path_for_config(config_path)
        if is_completed_result(result_path):
            print(f'>>> Skipping completed {config_path.name}')
            continue
        cleanup_for_config(config_path)
        run_config(config_path)
        cleanup_for_config(config_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
