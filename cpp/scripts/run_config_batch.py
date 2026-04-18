#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a batch of VecFlow experiment configs sequentially')
    parser.add_argument('--runner', required=True, help='Path to run_vecflow_experiment.py')
    parser.add_argument('--binary', required=True, help='Path to benchmark binary')
    parser.add_argument('--config-dir', required=True, help='Directory containing JSON configs')
    parser.add_argument('--pattern', default='*.json', help='Glob pattern within config-dir')
    parser.add_argument('--numa-node', type=int, default=1)
    parser.add_argument('--poll-seconds', type=float, default=1.0)
    parser.add_argument('--log-dir', help='Directory for per-config logs')
    parser.add_argument('--cleanup-indexes', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_dir = Path(args.config_dir)
    configs = sorted(config_dir.glob(args.pattern))
    if not configs:
        print(f'No configs matched {args.pattern} under {config_dir}', file=sys.stderr)
        return 1

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        cmd = [
            sys.executable,
            args.runner,
            '--binary', args.binary,
            '--config', str(config),
            '--numa-node', str(args.numa_node),
            '--poll-seconds', str(args.poll_seconds),
        ]
        if args.cleanup_indexes:
            cmd.append('--cleanup-indexes')
        if log_dir is not None:
            cmd.extend(['--log-file', str(log_dir / f'{config.stem}.log')])

        print(f'>>> Running {config.name}')
        print('>>> ' + shlex.join(cmd))
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            print(f'Config {config} failed with exit code {completed.returncode}', file=sys.stderr)
            return completed.returncode

    return 0


if __name__ == '__main__':
    sys.exit(main())
