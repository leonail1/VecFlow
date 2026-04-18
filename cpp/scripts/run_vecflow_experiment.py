#!/usr/bin/env python3
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run one VecFlow benchmark config with metric capture')
    parser.add_argument('--binary', required=True, help='Path to benchmark binary')
    parser.add_argument('--config', required=True, help='Path to benchmark config JSON')
    parser.add_argument('--numa-node', type=int, default=1, help='NUMA node for CPU and memory binding')
    parser.add_argument('--poll-seconds', type=float, default=1.0, help='Polling interval for external metrics')
    parser.add_argument('--cleanup-indexes', action='store_true', help='Delete generated index/cache artifacts after the run')
    parser.add_argument('--log-file', help='Optional file for benchmark stdout/stderr')
    return parser.parse_args()


def append_suffix_to_filename(filename: str, suffix: str) -> str:
    path = Path(filename)
    return str(path.with_name(f'{path.stem}{suffix}{path.suffix}'))


def read_proc_status_rss_bytes(pid: int) -> int:
    status_path = Path(f'/proc/{pid}/status')
    if not status_path.exists():
        return 0
    try:
        lines = status_path.read_text().splitlines()
    except (PermissionError, ProcessLookupError, FileNotFoundError):
        return 0
    for line in lines:
        if line.startswith('VmRSS:'):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return 0


def read_proc_io(pid: int) -> dict[str, int]:
    io_path = Path(f'/proc/{pid}/io')
    result = {'read_bytes': 0, 'write_bytes': 0, 'rchar': 0, 'wchar': 0}
    if not io_path.exists():
        return result
    try:
        lines = io_path.read_text().splitlines()
    except (PermissionError, ProcessLookupError, FileNotFoundError):
        return result
    for line in lines:
        key, value = line.split(':', 1)
        key = key.strip()
        if key in result:
            result[key] = int(value.strip())
    return result


def read_gpu_memory_mib(pid: int) -> int:
    cmd = [
        'nvidia-smi',
        '--query-compute-apps=pid,used_gpu_memory',
        '--format=csv,noheader,nounits',
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return 0
    peak = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 2:
            continue
        try:
            line_pid = int(parts[0])
            memory_mib = int(parts[1])
        except ValueError:
            continue
        if line_pid == pid:
            peak = max(peak, memory_mib)
    return peak


def read_pcie_throughput_mib_s() -> dict[str, float]:
    cmd = ['nvidia-smi', 'dmon', '-s', 't', '-c', '1']
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {'rxpci_mb_s': 0.0, 'txpci_mb_s': 0.0, 'samples': 0}

    peak_rx = 0.0
    peak_tx = 0.0
    samples = 0
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            _gpu_idx = int(parts[0])
            rx = float(parts[1])
            tx = float(parts[2])
        except ValueError:
            continue
        peak_rx = max(peak_rx, rx)
        peak_tx = max(peak_tx, tx)
        samples += 1
    return {'rxpci_mb_s': peak_rx, 'txpci_mb_s': peak_tx, 'samples': samples}


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


def augment_results(output_path: Path, external_metrics: dict[str, Any]) -> None:
    if not output_path.exists():
        return
    payload = json.loads(output_path.read_text())
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                item['external_metrics'] = external_metrics
    elif isinstance(payload, dict):
        payload['external_metrics'] = external_metrics
    output_path.write_text(json.dumps(payload, indent=2))


def cleanup_artifacts(artifacts: list[Path]) -> list[str]:
    removed = []
    for artifact in artifacts:
        if artifact.exists():
            artifact.unlink()
            removed.append(str(artifact))
    return removed


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    output_path = Path(config['output_json_file'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    cmd = [
        'numactl',
        f'--cpunodebind={args.numa_node}',
        f'--membind={args.numa_node}',
        args.binary,
        '--config',
        str(config_path),
    ]

    log_handle = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open('w')

    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=log_handle if log_handle is not None else None,
        stderr=subprocess.STDOUT if log_handle is not None else None,
    )
    peak_rss_bytes = 0
    peak_gpu_memory_mib = 0
    peak_rxpci_mb_s = 0.0
    peak_txpci_mb_s = 0.0
    pcie_samples = 0
    io_start = read_proc_io(process.pid)
    io_end = io_start.copy()

    try:
        while True:
            return_code = process.poll()
            peak_rss_bytes = max(peak_rss_bytes, read_proc_status_rss_bytes(process.pid))
            peak_gpu_memory_mib = max(peak_gpu_memory_mib, read_gpu_memory_mib(process.pid))
            pcie = read_pcie_throughput_mib_s()
            peak_rxpci_mb_s = max(peak_rxpci_mb_s, pcie['rxpci_mb_s'])
            peak_txpci_mb_s = max(peak_txpci_mb_s, pcie['txpci_mb_s'])
            pcie_samples += pcie['samples']
            io_end = read_proc_io(process.pid)
            if return_code is not None:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
        raise

    elapsed_seconds = time.time() - start_time
    external_metrics = {
        'command': cmd,
        'return_code': process.returncode,
        'elapsed_seconds': elapsed_seconds,
        'peak_rss_bytes': peak_rss_bytes,
        'peak_gpu_memory_mib': peak_gpu_memory_mib,
        'peak_rxpci_mb_s': peak_rxpci_mb_s,
        'peak_txpci_mb_s': peak_txpci_mb_s,
        'pcie_sample_count': pcie_samples,
        'read_bytes': max(0, io_end['read_bytes'] - io_start['read_bytes']),
        'write_bytes': max(0, io_end['write_bytes'] - io_start['write_bytes']),
        'rchar': max(0, io_end['rchar'] - io_start['rchar']),
        'wchar': max(0, io_end['wchar'] - io_start['wchar']),
    }
    augment_results(output_path, external_metrics)

    removed = []
    if args.cleanup_indexes:
        removed = cleanup_artifacts(collect_artifacts(config))
        external_metrics['removed_artifacts'] = removed
        augment_results(output_path, external_metrics)

    if log_handle is not None:
        log_handle.close()

    print(json.dumps({'external_metrics': external_metrics, 'removed_artifacts': removed}, indent=2))
    return process.returncode


if __name__ == '__main__':
    sys.exit(main())
