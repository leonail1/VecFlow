#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot VecFlow strict QPS and cache hit-rate time series')
    parser.add_argument('--series', action='append', required=True,
                        help='Series in label=path.json form; can be passed multiple times')
    parser.add_argument('--algorithm', default=None,
                        help='Optional algorithm filter when input JSON is a result array')
    parser.add_argument('--itopk', type=int, default=None,
                        help='Optional itopk filter when input JSON is a result array')
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def load_series(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_series_arg(raw: str) -> tuple[str, Path]:
    if '=' not in raw:
        raise SystemExit(f'Invalid --series value: {raw}; expected label=path')
    label, path = raw.split('=', 1)
    return label, Path(path)


def cumulative_metrics(sample: dict) -> dict[str, float]:
    counters = sample['cache_counters']
    return {
        'queries_completed': float(sample['queries_completed']),
        'graph_hbm': float(counters['phoenix_graph']['hbm_hits']),
        'graph_dram': float(counters['phoenix_graph']['dram_hits']),
        'graph_ssd': float(counters['phoenix_graph']['ssd_loads']),
        'dataset_hbm': float(counters['phoenix_dataset']['hbm_hits']),
        'dataset_dram': float(counters['phoenix_dataset']['dram_hits']),
        'dataset_ssd': float(counters['phoenix_dataset']['ssd_loads']),
        'bfs_hbm': float(counters['tiered_bfs']['hbm_hits']),
        'bfs_dram': float(counters['tiered_bfs']['dram_hits']),
        'bfs_ssd': float(counters['tiered_bfs']['ssd_loads']),
    }


def select_entry(payload: dict | list, algorithm: str | None, itopk: int | None) -> dict:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, list) or not payload:
        raise SystemExit('Input JSON must be an object or a non-empty result array')
    for entry in payload:
        if algorithm is not None and entry.get('algorithm') != algorithm:
            continue
        if itopk is not None and int(entry.get('itopk', -1)) != itopk:
            continue
        return entry
    raise SystemExit('No result entry matched --algorithm/--itopk filters')


def resolve_samples(payload: dict | list, algorithm: str | None, itopk: int | None) -> list[dict]:
    entry = select_entry(payload, algorithm, itopk)
    strict_samples = entry.get('strict_qps_time_series')
    if isinstance(strict_samples, list) and strict_samples:
        return strict_samples
    legacy_samples = entry.get('samples')
    if isinstance(legacy_samples, list) and legacy_samples:
        return legacy_samples
    raise SystemExit('No strict_qps_time_series or legacy samples found in input JSON')


def bucketize(samples: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for bucket in range(1, len(samples)):
        s0 = samples[bucket - 1]
        s1 = samples[bucket]
        t0 = float(s0['elapsed_seconds'])
        t1 = float(s1['elapsed_seconds'])
        dt = t1 - t0
        if dt <= 0:
            continue
        m0 = cumulative_metrics(s0)
        m1 = cumulative_metrics(s1)
        dq = m1['queries_completed'] - m0['queries_completed']
        qps = dq / dt

        graph_hbm = m1['graph_hbm'] - m0['graph_hbm']
        graph_dram = m1['graph_dram'] - m0['graph_dram']
        graph_ssd = m1['graph_ssd'] - m0['graph_ssd']
        dataset_hbm = m1['dataset_hbm'] - m0['dataset_hbm']
        dataset_dram = m1['dataset_dram'] - m0['dataset_dram']
        dataset_ssd = m1['dataset_ssd'] - m0['dataset_ssd']
        bfs_hbm = m1['bfs_hbm'] - m0['bfs_hbm']
        bfs_dram = m1['bfs_dram'] - m0['bfs_dram']
        bfs_ssd = m1['bfs_ssd'] - m0['bfs_ssd']

        combined_hbm = graph_hbm + dataset_hbm + bfs_hbm
        combined_dram = graph_dram + dataset_dram + bfs_dram
        combined_ssd = graph_ssd + dataset_ssd + bfs_ssd
        combined_total = combined_hbm + combined_dram + combined_ssd
        bfs_total = bfs_hbm + bfs_dram + bfs_ssd
        graph_total = graph_hbm + graph_dram + graph_ssd
        dataset_total = dataset_hbm + dataset_dram + dataset_ssd

        rows.append({
            'bucket': bucket,
            't0_seconds': t0,
            't1_seconds': t1,
            'delta_seconds': dt,
            'qps': qps,
            'combined_hbm_rate': (combined_hbm / combined_total) if combined_total > 0 else 0.0,
            'combined_dram_rate': (combined_dram / combined_total) if combined_total > 0 else 0.0,
            'combined_ssd_rate': (combined_ssd / combined_total) if combined_total > 0 else 0.0,
            'graph_hbm_rate': (graph_hbm / graph_total) if graph_total > 0 else 0.0,
            'graph_dram_rate': (graph_dram / graph_total) if graph_total > 0 else 0.0,
            'dataset_hbm_rate': (dataset_hbm / dataset_total) if dataset_total > 0 else 0.0,
            'dataset_dram_rate': (dataset_dram / dataset_total) if dataset_total > 0 else 0.0,
            'bfs_hbm_rate': (bfs_hbm / bfs_total) if bfs_total > 0 else 0.0,
            'bfs_dram_rate': (bfs_dram / bfs_total) if bfs_total > 0 else 0.0,
            'combined_hbm_hits': combined_hbm,
            'combined_dram_hits': combined_dram,
            'combined_ssd_loads': combined_ssd,
            'bfs_hbm_hits': bfs_hbm,
            'bfs_dram_hits': bfs_dram,
            'bfs_ssd_loads': bfs_ssd,
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_one(label: str, rows: list[dict], output_png: Path) -> None:
    import matplotlib.pyplot as plt

    xs = [row['t1_seconds'] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(xs, [row['qps'] for row in rows], label=label, color='tab:blue')
    axes[0].set_ylabel('QPS')
    axes[0].set_title(f'{label}: per-second QPS')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, [row['combined_hbm_rate'] for row in rows], label='HBM hit rate', color='tab:green')
    axes[1].plot(xs, [row['combined_dram_rate'] for row in rows], label='DRAM hit rate', color='tab:orange')
    axes[1].plot(xs, [row['combined_ssd_rate'] for row in rows], label='SSD load share', color='tab:red')
    axes[1].set_xlabel('Elapsed seconds')
    axes[1].set_ylabel('Share')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title('Combined per-second cache outcome share')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')

    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_comparison(series_rows: dict[str, list[dict]], output_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for label, rows in series_rows.items():
        xs = [row['t1_seconds'] for row in rows]
        ax.plot(xs, [row['qps'] for row in rows], label=label)
    ax.set_xlabel('Elapsed seconds')
    ax.set_ylabel('QPS')
    ax.set_title('Per-second QPS comparison')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict]] = {}
    for raw in args.series:
        label, path = parse_series_arg(raw)
        payload = load_series(path)
        samples = resolve_samples(payload, args.algorithm, args.itopk)
        rows = bucketize(samples)
        all_rows[label] = rows
        write_csv(output_dir / f'{label}_per_second.csv', rows)
        plot_one(label, rows, output_dir / f'{label}_qps_hit_rates.png')

    if len(all_rows) >= 2:
        plot_comparison(all_rows, output_dir / 'qps_comparison.png')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
