#!/usr/bin/env python3
"""Reuse existing DiskANN memory index and sweep single-label filtered recall.

This avoids rebuilding index and focuses on node6-style single filter_label evaluation.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

COARSE_LS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000]
TARGET_SELECTIVITIES = [0.004, 0.005, 0.01, 0.02, 0.05, 0.099, 0.192]


def run(cmd: list[str], log: Path | None = None, check: bool = True) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(out)
    if check and p.returncode != 0:
        raise RuntimeError(f"cmd failed ({p.returncode}): {' '.join(cmd)}\n{out[-2000:]}")
    return out


def parse_rows(output: str):
    pat = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
    rows = []
    for line in output.splitlines():
        m = pat.match(line)
        if not m:
            continue
        l, qps, cmps, lat, p99, rec = m.groups()
        rows.append({
            "L": int(l),
            "qps": float(qps),
            "avg_cmps": float(cmps),
            "latency_us": float(lat),
            "p99_latency_us": float(p99),
            "recall": float(rec),
        })
    return rows


def read_u8bin(path: Path):
    with path.open("rb") as f:
        n, d = struct.unpack("II", f.read(8))
        data = f.read(n * d)
    return n, d, data


def write_u8bin(path: Path, n: int, d: int, rows: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("II", n, d))
        f.write(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/lzg/VecFlow-develop")
    ap.add_argument("--index-prefix", default="/home/lzg/VecFlow-develop/experiment_results/three_way_comparison/diskann_index/yfcc10m")
    ap.add_argument("--threads-search", type=int, default=1)
    ap.add_argument("--recall-target", type=float, default=98.0)
    ap.add_argument("--nq", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.root)
    ds = root / "vecflow/datasets/yfcc10M"
    out = root / "experiment_results/three_way_comparison/filteredvamana_reuse_index_repro"
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)

    search_bin = root / "thirdparty/DiskANN/build/apps/search_memory_index"
    gt_bin = root / "thirdparty/DiskANN/build/apps/utils/compute_groundtruth_for_filters"

    index_prefix = Path(args.index_prefix)
    if not index_prefix.exists():
        raise SystemExit(f"Missing index prefix: {index_prefix}")

    base_file = ds / "base.10M.u8bin"
    query_file = ds / "query.public.single_label.u8bin"
    base_labels = ds / "base_labels_diskann.txt"
    query_filters = ds / "query_filters_diskann.txt"

    # map labels to counts and query ids
    base_counts = Counter()
    with base_labels.open() as f:
        for line in f:
            s = line.strip()
            if not s or s == "-1":
                continue
            for tok in s.split(","):
                if tok:
                    base_counts[int(tok)] += 1

    qids_by_label = defaultdict(list)
    with query_filters.open() as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s or s == "-1":
                continue
            qids_by_label[int(s.split(",")[0])].append(i)

    _, dim, qbytes = read_u8bin(query_file)

    results = []
    for target in TARGET_SELECTIVITIES:
        # pick nearest selectivity label with enough queries
        best = None
        for lid, cnt in base_counts.items():
            qn = len(qids_by_label.get(lid, []))
            if qn < args.nq:
                continue
            sel = cnt / 10_000_000
            d = abs(sel - target)
            if best is None or d < best[0]:
                best = (d, lid, sel, cnt, qn)
        if best is None:
            continue
        _, label, sel, cnt, qn_total = best
        qids = qids_by_label[label][: args.nq]

        # write query subset and one-label filter file
        tag = str(target).replace(".", "p")
        sub_q = out / f"cache/query_sel{tag}_n{len(qids)}.u8bin"
        sub_f = out / f"cache/query_sel{tag}_n{len(qids)}.filters.txt"
        sub_gt = out / f"cache/gt_sel{tag}_n{len(qids)}.bin"

        buf = bytearray()
        for qid in qids:
            st = qid * dim
            buf.extend(qbytes[st: st + dim])
        write_u8bin(sub_q, len(qids), dim, bytes(buf))
        sub_f.write_text("\n".join([str(label)] * len(qids)) + "\n")

        if not sub_gt.exists():
            gt_cmd = [
                "numactl", "--cpunodebind=1", "--membind=1",
                str(gt_bin),
                "--data_type", "uint8",
                "--dist_fn", "l2",
                "--base_file", str(base_file),
                "--query_file", str(sub_q),
                "--label_file", str(base_labels),
                "--filter_label_file", str(sub_f),
                "--gt_file", str(sub_gt),
                "--K", "10",
            ]
            run(gt_cmd, logs / f"gt_sel{tag}.log", check=True)

        search_cmd = [
            "numactl", "--cpunodebind=1", "--membind=1",
            str(search_bin),
            "--data_type", "uint8",
            "--dist_fn", "l2",
            "--index_path_prefix", str(index_prefix),
            "--query_file", str(sub_q),
            "--gt_file", str(sub_gt),
            "--filter_label", str(label),
            "--label_type", "uint",
            "-K", "10",
            "-T", str(args.threads_search),
            "-L", *[str(v) for v in COARSE_LS],
            "--result_path", str(out / f"results/sel{tag}"),
        ]
        # Some filtered queries can have fewer than K candidates; DiskANN may return
        # a non-zero code after printing usable metrics rows. Parse whatever it emitted.
        out_text = run(search_cmd, logs / f"search_sel{tag}.log", check=False)
        rows = parse_rows(out_text)
        best_row = None
        for r in rows:
            if r["recall"] >= args.recall_target:
                best_row = r
                break
        if best_row is None and rows:
            best_row = max(rows, key=lambda x: x["recall"])

        if best_row:
            results.append({
                "selectivity_target": target,
                "selectivity_actual": sel,
                "label": label,
                "base_count": cnt,
                "query_count": len(qids),
                "threads": args.threads_search,
                "best_L": best_row["L"],
                "recall": best_row["recall"],
                "qps": best_row["qps"],
                "latency_us": best_row["latency_us"],
                "p99_latency_us": best_row["p99_latency_us"],
            })
            print(f"sel={target:.3f} actual={sel:.4f} label={label} L={best_row['L']} recall={best_row['recall']:.2f} qps={best_row['qps']:.2f}")

    out_csv = out / "yfcc10m_filteredvamana_reuse_index_repro.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "selectivity_target", "selectivity_actual", "label", "base_count", "query_count", "threads",
            "best_L", "recall", "qps", "latency_us", "p99_latency_us",
        ])
        writer.writeheader()
        writer.writerows(results)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
