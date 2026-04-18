#!/usr/bin/env python3
import json
import math
from pathlib import Path
import bisect

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/lzg/VecFlow-develop/experiment_results/three_way_comparison')
TS = ROOT / 'develop_timeseries_label29_n200.json'
OUT = ROOT / 'three_way_qps_comparison_label29_n200.png'
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


def per_second_qps(times, qdone):
    total = times[-1]
    n = int(math.floor(total))
    xs, ys = [], []
    for b in range(n):
        t0, t1 = float(b), float(b + 1)
        q0 = interp(times, qdone, t0)
        q1 = interp(times, qdone, t1)
        xs.append(t1)
        ys.append(q1 - q0)
    return xs, ys


def main():
    times, qdone = load_timeseries(TS)
    xs, ys = per_second_qps(times, qdone)
    dev_avg = (qdone[-1] / times[-1]) if times[-1] > 0 else 0.0

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(xs, ys, color='#1f77b4', linewidth=1.2, label=f'VecFlow-develop per-second (avg={dev_avg:,.0f})')
    ax.axhline(ORIG_QPS, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Original VecFlow (subset QPS={ORIG_QPS:,.0f})')
    ax.axhline(FILTERED_QPS, color='#d62728', linestyle='-.', linewidth=2,
               label=f'FilteredVamana @recall>=98% (QPS={FILTERED_QPS:,.2f})')

    ax.set_title('YFCC-10M label=29 subset (n=200) - QPS vs time')
    ax.set_xlabel('Elapsed seconds')
    ax.set_ylabel('QPS')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)

    with CSV.open('w') as f:
        f.write('system,qps,notes\n')
        f.write(f'Original VecFlow,{ORIG_QPS},label29_n200 subset\n')
        f.write(f'FilteredVamana,{FILTERED_QPS},single-filter label29 recall>=98 point\n')
        f.write(f'VecFlow-develop(avg),{dev_avg},label29_n200 subset timeseries\n')

    print('saved', OUT)
    print('saved', CSV)


if __name__ == '__main__':
    main()
