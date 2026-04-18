#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/lzg/VecFlow-develop')
BUILD_DIR = ROOT / 'vecflow/examples/cpp/build'
BINARY = BUILD_DIR / 'VECFLOW_QPS_TIMESERIES'
GENERATOR = ROOT / 'cpp/scripts/generate_yfcc_timeseries_configs.py'
PLOTTER = ROOT / 'cpp/scripts/plot_vecflow_qps_timeseries.py'
CONFIG_DIR = ROOT / 'cpp/scripts/generated_timeseries_configs'

SERIES = [
    ('q61626', CONFIG_DIR / 'yfcc_timeseries_q61626_it64.json', Path('/tmp/yfcc_timeseries_q61626_it64.json')),
    ('q600k', CONFIG_DIR / 'yfcc_timeseries_q600k_it64.json', Path('/tmp/yfcc_timeseries_q600k_it64.json')),
    ('q6000k', CONFIG_DIR / 'yfcc_timeseries_q6000k_it64.json', Path('/tmp/yfcc_timeseries_q6000k_it64.json')),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run YFCC time-series diagnostic after the main suites finish')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--output-dir', default='/tmp/yfcc_qps_timeseries_plots')
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print('>>>', ' '.join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    run([sys.executable, str(GENERATOR)])
    if not args.skip_build:
        run([
            'numactl', '--cpunodebind=1', '--membind=1',
            'cmake', '--build', str(BUILD_DIR), '--target', 'VECFLOW_QPS_TIMESERIES', '-j54',
        ])
    for _label, config_path, _result_path in SERIES:
        run([
            'numactl', '--cpunodebind=1', '--membind=1',
            str(BINARY), '--config', str(config_path),
        ])
    plot_cmd = [sys.executable, str(PLOTTER), '--output-dir', args.output_dir]
    for label, _config_path, result_path in SERIES:
        plot_cmd.extend(['--series', f'{label}={result_path}'])
    run(plot_cmd)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
