#!/usr/bin/env python3
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a filtered DiskANN experiment with metric capture')
    parser.add_argument('--diskann-build-dir', required=True, help='DiskANN build directory containing apps/')
    parser.add_argument('--data-file', required=True, help='Base vectors in DiskANN .bin format')
    parser.add_argument('--query-file', required=True, help='Query vectors in DiskANN .bin format')
    parser.add_argument('--label-file', required=True, help='Base labels in DiskANN txt format')
    parser.add_argument('--query-filters-file', required=True, help='One filter string per query')
    parser.add_argument('--index-prefix', required=True, help='Output prefix for DiskANN index files')
    parser.add_argument('--gt-file', required=True, help='Output path for filtered ground truth')
    parser.add_argument('--result-prefix', required=True, help='Output prefix for search result files')
    parser.add_argument('--output-json', required=True, help='Output JSON summary path')
    parser.add_argument('--data-type', default='uint8', choices=['float', 'int8', 'uint8'])
    parser.add_argument('--dist-fn', default='l2', choices=['l2', 'mips', 'fast_l2', 'cosine'])
    parser.add_argument('--numa-node', type=int, default=1)
    parser.add_argument('--threads', type=int, default=52)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--build-r', type=int, default=64)
    parser.add_argument('--build-l', type=int, default=100)
    parser.add_argument('--filtered-lbuild', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1.2)
    parser.add_argument('--search-ls', nargs='+', type=int, required=True, help='Search L values')
    parser.add_argument('--label-type', default='uint', choices=['uint', 'uint32', 'ushort', 'uint16'])
    parser.add_argument('--poll-seconds', type=float, default=1.0)
    parser.add_argument('--log-dir', help='Optional directory for build/gt/search logs')
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--cleanup-index', action='store_true')
    parser.add_argument('--cleanup-results', action='store_true')
    return parser.parse_args()


def read_proc_status_rss_bytes(pid: int) -> int:
    status_path = Path(f'/proc/{pid}/status')
    if not status_path.exists():
        return 0
    try:
        for line in status_path.read_text().splitlines():
            if line.startswith('VmRSS:'):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except (PermissionError, ProcessLookupError, FileNotFoundError):
        return 0
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


def normalize_label_type(value: str) -> str:
    if value == 'uint32':
        return 'uint'
    if value == 'uint16':
        return 'ushort'
    return value


def run_command(cmd: list[str], poll_seconds: float, log_path: Path | None, capture_output: bool) -> dict:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open('w')
    else:
        log_handle = None

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    peak_rss_bytes = 0
    io_start = read_proc_io(process.pid)
    io_end = io_start.copy()
    captured_lines: list[str] = []
    start_time = time.time()

    try:
        while True:
            line = process.stdout.readline() if process.stdout is not None else ''
            if line:
                if log_handle is not None:
                    log_handle.write(line)
                    log_handle.flush()
                if capture_output:
                    captured_lines.append(line)
            elif process.poll() is not None:
                break
            peak_rss_bytes = max(peak_rss_bytes, read_proc_status_rss_bytes(process.pid))
            io_end = read_proc_io(process.pid)
            time.sleep(poll_seconds)

        if process.stdout is not None:
            remainder = process.stdout.read()
            if remainder:
                if log_handle is not None:
                    log_handle.write(remainder)
                    log_handle.flush()
                if capture_output:
                    captured_lines.extend(remainder.splitlines(keepends=True))
        process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
        raise
    finally:
        if log_handle is not None:
            log_handle.close()

    elapsed_seconds = time.time() - start_time
    return {
        'command': cmd,
        'return_code': process.returncode,
        'elapsed_seconds': elapsed_seconds,
        'peak_rss_bytes': peak_rss_bytes,
        'read_bytes': max(0, io_end['read_bytes'] - io_start['read_bytes']),
        'write_bytes': max(0, io_end['write_bytes'] - io_start['write_bytes']),
        'rchar': max(0, io_end['rchar'] - io_start['rchar']),
        'wchar': max(0, io_end['wchar'] - io_start['wchar']),
        'stdout': ''.join(captured_lines) if capture_output else '',
        'log_file': str(log_path) if log_path is not None else None,
    }


def parse_search_table(stdout: str) -> list[dict]:
    rows: list[dict] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            row = {
                'L': int(parts[0]),
                'qps': float(parts[1]),
            }
            if len(parts) >= 6:
                row['avg_cmps'] = float(parts[2])
                row['mean_latency_us'] = float(parts[3])
                row['p999_latency_us'] = float(parts[4])
                row['recall_at_k'] = float(parts[5]) if len(parts) >= 6 else None
            else:
                row['mean_latency_us'] = float(parts[2])
                row['p999_latency_us'] = float(parts[3])
            if len(parts) >= 7:
                extra_recalls = []
                for value in parts[5:]:
                    extra_recalls.append(float(value))
                row['recalls'] = extra_recalls
                row['recall_at_k'] = extra_recalls[-1]
            rows.append(row)
        except ValueError:
            continue
    return rows


def existing_index_files(index_prefix: Path) -> list[Path]:
    parent = index_prefix.parent
    stem = index_prefix.name
    return sorted(parent.glob(stem + '*'))


def cleanup_paths(paths: list[Path]) -> list[str]:
    removed: list[str] = []
    for path in paths:
        if path.exists():
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def main() -> int:
    args = parse_args()
    diskann_build_dir = Path(args.diskann_build_dir)
    build_bin = diskann_build_dir / 'apps/build_memory_index'
    search_bin = diskann_build_dir / 'apps/search_memory_index'
    gt_bin = diskann_build_dir / 'apps/utils/compute_groundtruth_for_filters'
    for binary in [build_bin, search_bin, gt_bin]:
        if not binary.exists():
            raise SystemExit(f'Missing DiskANN binary: {binary}')

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else None
    label_type = normalize_label_type(args.label_type)

    def with_numa(cmd: list[str]) -> list[str]:
        return ['numactl', f'--cpunodebind={args.numa_node}', f'--membind={args.numa_node}', *cmd]

    build_cmd = with_numa([
        str(build_bin),
        '--data_type', args.data_type,
        '--dist_fn', args.dist_fn,
        '--data_path', args.data_file,
        '--index_path_prefix', args.index_prefix,
        '-R', str(args.build_r),
        '-L', str(args.build_l),
        '--FilteredLbuild', str(args.filtered_lbuild),
        '--alpha', str(args.alpha),
        '-T', str(args.threads),
        '--label_file', args.label_file,
        '--label_type', label_type,
    ])

    gt_cmd = with_numa([
        str(gt_bin),
        '--data_type', args.data_type,
        '--dist_fn', args.dist_fn,
        '--base_file', args.data_file,
        '--query_file', args.query_file,
        '--label_file', args.label_file,
        '--filter_label_file', args.query_filters_file,
        '--gt_file', args.gt_file,
        '--K', str(args.k),
    ])

    search_cmd = with_numa([
        str(search_bin),
        '--data_type', args.data_type,
        '--dist_fn', args.dist_fn,
        '--index_path_prefix', args.index_prefix,
        '--query_file', args.query_file,
        '--query_filters_file', args.query_filters_file,
        '--gt_file', args.gt_file,
        '--label_type', label_type,
        '-K', str(args.k),
        '-T', str(args.threads),
        '-L', *[str(v) for v in args.search_ls],
        '--result_path', args.result_prefix,
    ])

    build_needed = args.force_rebuild or not existing_index_files(Path(args.index_prefix))
    gt_needed = args.force_rebuild or not Path(args.gt_file).exists()

    summary = {
        'diskann_build_dir': str(diskann_build_dir),
        'data_file': args.data_file,
        'query_file': args.query_file,
        'label_file': args.label_file,
        'query_filters_file': args.query_filters_file,
        'index_prefix': args.index_prefix,
        'gt_file': args.gt_file,
        'result_prefix': args.result_prefix,
        'search_ls': args.search_ls,
        'build': {'skipped': not build_needed, 'metrics': None},
        'groundtruth': {'skipped': not gt_needed, 'metrics': None},
        'search': {'metrics': None, 'rows': []},
        'cleanup': {'removed': []},
    }

    if build_needed:
        build_metrics = run_command(
            build_cmd,
            args.poll_seconds,
            log_dir / 'build.log' if log_dir else None,
            capture_output=False,
        )
        summary['build']['metrics'] = build_metrics
        if build_metrics['return_code'] != 0:
            output_json.write_text(json.dumps(summary, indent=2))
            return build_metrics['return_code']

    if gt_needed:
        gt_metrics = run_command(
            gt_cmd,
            args.poll_seconds,
            log_dir / 'groundtruth.log' if log_dir else None,
            capture_output=False,
        )
        summary['groundtruth']['metrics'] = gt_metrics
        if gt_metrics['return_code'] != 0:
            output_json.write_text(json.dumps(summary, indent=2))
            return gt_metrics['return_code']

    search_metrics = run_command(
        search_cmd,
        args.poll_seconds,
        log_dir / 'search.log' if log_dir else None,
        capture_output=True,
    )
    summary['search']['metrics'] = search_metrics
    summary['search']['rows'] = parse_search_table(search_metrics['stdout'])

    if args.cleanup_index:
        summary['cleanup']['removed'].extend(cleanup_paths(existing_index_files(Path(args.index_prefix))))
    if args.cleanup_results:
        result_prefix = Path(args.result_prefix)
        summary['cleanup']['removed'].extend(cleanup_paths(sorted(result_prefix.parent.glob(result_prefix.name + '*'))))
        summary['cleanup']['removed'].extend(cleanup_paths([Path(args.gt_file)]))

    output_json.write_text(json.dumps(summary, indent=2))
    return search_metrics['return_code']


if __name__ == '__main__':
    sys.exit(main())
