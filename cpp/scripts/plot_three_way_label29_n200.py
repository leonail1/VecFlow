#!/usr/bin/env python3
import bisect
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/lzg/VecFlow-develop/experiment_results/three_way_comparison')
TS = ROOT / 'develop_timeseries_label29_n200.json'
OUT = ROOT / 'three_way_qps_comparison_label29_n200.png'
OUT_CORRECTED = ROOT / 'three_way_qps_comparison_label29_n200_corrected.png'
CSV = ROOT / 'three_way_summary_label29_n200.csv'

ORIG_QPS = 2131435.17
FILTERED_QPS = 30.01  # from recall>=98% point (selectivity~19.2%, threads=1)


def load_timeseries(path: Path):
    payload = json.loads(path.read_text())
    samples = payload['samples']
    times = [float(s['elapsed_seconds']) for s in samples]
    qdone = [float(s['queries_completed']) for s in samples]
    return times, qdone


def interp(times, vals, t):
    if t <= times[0]:
        return vals[0]
    if t >= times[-1]:
        return vals[-1]
    i = bisect.bisect_right(times, t)
    t0, t1 = times[i - 1], times[i]
    v0, v1 = vals[i - 1], vals[i]
    if t1 <= t0:
        return v1
    a = (t - t0) / (t1 - t0)
    return v0 + (v1 - v0) * a


def bucket_qps(times, qdone, bucket_seconds: float = 0.5):
    total = times[-1]
    n = int(math.floor(total / bucket_seconds))
    xs, ys = [], []
    for b in range(n):
        t0 = b * bucket_seconds
        t1 = (b + 1) * bucket_seconds
        q0 = interp(times, qdone, t0)
        q1 = interp(times, qdone, t1)
        xs.append(t1)
        ys.append((q1 - q0) / bucket_seconds)
    return xs, ys


def main():
    times, qdone = load_timeseries(TS)
    xs, ys = bucket_qps(times, qdone, bucket_seconds=0.5)
    dev_avg = (qdone[-1] / times[-1]) if times[-1] > 0 else 0.0

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(xs, ys, color='#1f77b4', linewidth=1.2, label=f'VecFlow-develop instant QPS (avg={dev_avg:,.0f})')
    ax.axhline(ORIG_QPS, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Original VecFlow (subset QPS={ORIG_QPS:,.0f})')
    ax.axhline(FILTERED_QPS, color='#d62728', linestyle='-.', linewidth=2,
               label=f'FilteredVamana @recall>=98% (QPS={FILTERED_QPS:,.2f})')

    ax.set_title('YFCC-10M label=29 subset (n=200) - QPS vs time (linear)')
    ax.set_xlabel('Elapsed seconds')
    ax.set_ylabel('QPS')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)

    # Corrected visibility for multi-order magnitude differences.
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(xs, ys, color='#1f77b4', linewidth=1.0, label='VecFlow-develop instant QPS')
    ax1.axhline(ORIG_QPS, color='#ff7f0e', linestyle='--', linewidth=1.8, label='Original VecFlow baseline')
    ax1.axhline(FILTERED_QPS, color='#d62728', linestyle='-.', linewidth=1.8, label='FilteredVamana baseline')
    ax1.set_ylabel('QPS (linear)')
    ax1.set_title('QPS vs time (linear scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    ax2.plot(xs, [max(v, 1e-6) for v in ys], color='#1f77b4', linewidth=1.0, label='VecFlow-develop instant QPS')
    ax2.axhline(max(ORIG_QPS, 1e-6), color='#ff7f0e', linestyle='--', linewidth=1.8, label='Original VecFlow baseline')
    ax2.axhline(max(FILTERED_QPS, 1e-6), color='#d62728', linestyle='-.', linewidth=1.8, label='FilteredVamana baseline')
    ax2.set_yscale('log')
    ax2.set_ylabel('QPS (log10)')
    ax2.set_xlabel('Elapsed seconds')
    ax2.set_title('QPS vs time (log scale, corrected visibility)')
    ax2.grid(True, alpha=0.3, which='both')

    fig2.tight_layout()
    fig2.savefig(OUT_CORRECTED, dpi=160)

    with CSV.open('w') as f:
        f.write('system,qps,notes\n')
        f.write(f'Original VecFlow,{ORIG_QPS},label29_n200 subset\n')
        f.write(f'FilteredVamana,{FILTERED_QPS},single-filter label29 recall>=98 point\n')
        f.write(f'VecFlow-develop(avg),{dev_avg},label29_n200 subset timeseries\n')

    print('saved', OUT)
    print('saved', OUT_CORRECTED)
    print('saved', CSV)


if __name__ == '__main__':
    main()
