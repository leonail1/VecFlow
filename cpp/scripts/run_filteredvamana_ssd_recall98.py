#!/usr/bin/env python3
"""Reproduce node6-style FilteredVamana SSD sweep on YFCC-10M.

- Build SSD filtered index with node6-like params
- For each target selectivity, pick one representative label
- Build query subset (default 1000 queries) for that label
- Compute filtered GT
- Sweep L on search_disk_index and pick min L reaching recall target
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

TARGET_SELECTIVITIES = [0.004, 0.005, 0.01, 0.02, 0.05, 0.099, 0.192]
COARSE_LS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000]


def run(cmd: list[str], log_path: Path | None = None, check: bool = True) -> str:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines: list[str] = []
    with (log_path.open("w") if log_path else open("/dev/null", "w")) as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            if log_path:
                lf.write(line)
                lf.flush()
        proc.wait()
    out = "".join(output_lines)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{out[-2000:]}")
    return out


def read_u8bin(path: Path):
    with path.open("rb") as f:
        n, d = struct.unpack("II", f.read(8))
        data = f.read(n * d)
    return n, d, data


def write_u8bin(path: Path, n: int, d: int, rows: bytes):
    with path.open("wb") as f:
        f.write(struct.pack("II", n, d))
        f.write(rows)


def parse_search_rows(output: str) -> list[dict]:
    rows = []
    pat = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
    for line in output.splitlines():
        m = pat.match(line)
        if not m:
            continue
        l, qps, cmps, lat, p99, rec = m.groups()
        rows.append(
            {
                "L": int(l),
                "qps": float(qps),
                "avg_cmps": float(cmps),
                "latency_us": float(lat),
                "p99_latency_us": float(p99),
                "recall": float(rec),
            }
        )
    return rows


def build_label_maps(base_labels_txt: Path, query_filters_txt: Path):
    base_cnt = Counter()
    with base_labels_txt.open() as f:
        for line in f:
            s = line.strip()
            if not s or s == "-1":
                continue
            for tok in s.split(","):
                if tok:
                    base_cnt[int(tok)] += 1

    query_ids_by_label: dict[int, list[int]] = defaultdict(list)
    with query_filters_txt.open() as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s or s == "-1":
                continue
            label = int(s.split(",")[0])
            query_ids_by_label[label].append(i)
    return base_cnt, query_ids_by_label


def pick_label_for_selectivity(target: float, base_cnt: Counter, query_ids_by_label: dict[int, list[int]], base_n: int, min_queries: int):
    best = None
    best_score = None
    for label, count in base_cnt.items():
        qn = len(query_ids_by_label.get(label, []))
        if qn < min_queries:
            continue
        sel = count / base_n
        score = abs(sel - target)
        if best_score is None or score < best_score:
            best_score = score
            best = (label, count, sel, qn)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/lzg/VecFlow-develop")
    ap.add_argument("--threads-build", type=int, default=52)
    ap.add_argument("--threads-search", type=int, default=1)
    ap.add_argument("--recall-target", type=float, default=98.0)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--min-queries-per-label", type=int, default=100)
    ap.add_argument("--search-dram-budget-gb", type=int, default=2)
    ap.add_argument("--build-dram-budget-gb", type=int, default=64)
    ap.add_argument("--pq-disk-bytes", type=int, default=0)
    ap.add_argument("--build-pq-bytes", type=int, default=16)
    ap.add_argument("--force-build", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    ds = root / "vecflow/datasets/yfcc10M"
    out = root / "experiment_results/three_way_comparison/filteredvamana_ssd_repro"
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)

    build_bin = root / "thirdparty/DiskANN/build/apps/build_disk_index"
    search_bin = root / "thirdparty/DiskANN/build/apps/search_disk_index"
    gt_bin = root / "thirdparty/DiskANN/build/apps/utils/compute_groundtruth_for_filters"

    base_file = ds / "base.10M.u8bin"
    query_file = ds / "query.public.single_label.u8bin"
    base_labels = ds / "base_labels_diskann.txt"
    query_filters = ds / "query_filters_diskann.txt"

    index_prefix = out / "indices/yfcc10m_filteredvamana_ssd_R64_L100_F200_a1p2"

    # Build SSD index
    required_files = [
        Path(str(index_prefix) + "_disk.index"),
        Path(str(index_prefix) + "_pq_compressed.bin"),
    ]
    build_needed = args.force_build or any(not p.exists() for p in required_files)
    if build_needed:
        cmd = [
            "numactl", "--cpunodebind=1", "--membind=1",
            str(build_bin),
            "--data_type", "uint8",
            "--dist_fn", "l2",
            "--index_path_prefix", str(index_prefix),
            "--data_path", str(base_file),
            "-T", str(args.threads_build),
            "-R", "64",
            "-L", "100",
            "--FilteredLbuild", "200",
            "-B", str(args.search_dram_budget_gb),
            "-M", str(args.build_dram_budget_gb),
            "--PQ_disk_bytes", str(args.pq_disk_bytes),
            "--build_PQ_bytes", str(args.build_pq_bytes),
            "-F", "0",
            "--label_file", str(base_labels),
            "--label_type", "uint",
        ]
        run(cmd, logs / "build_ssd.log", check=True)

    # Prepare label maps
    base_n, dim, q_data = read_u8bin(query_file)
    base_vecs, _, _ = read_u8bin(base_file)
    base_cnt, qids_by_label = build_label_maps(base_labels, query_filters)

    csv_rows = []
    for target in TARGET_SELECTIVITIES:
        picked = pick_label_for_selectivity(target, base_cnt, qids_by_label, base_vecs, args.min_queries_per_label)
        if picked is None:
            continue
        label, base_count, actual_sel, qn = picked
        ids = qids_by_label[label][: args.nq]
        nq = len(ids)
        if nq == 0:
            continue

        # Build query subset
        row_size = dim
        sub = bytearray()
        for qid in ids:
            st = qid * row_size
            sub.extend(q_data[st: st + row_size])

        sel_tag = str(target).replace(".", "p")
        sub_q = out / f"cache/query_sel{sel_tag}_n{nq}.u8bin"
        sub_f = out / f"cache/query_sel{sel_tag}_n{nq}.filters.txt"
        sub_gt = out / f"cache/gt_sel{sel_tag}_n{nq}.bin"
        write_u8bin(sub_q, nq, dim, bytes(sub))
        sub_f.parent.mkdir(parents=True, exist_ok=True)
        sub_f.write_text("\n".join([str(label)] * nq) + "\n")

        # GT
        if not sub_gt.exists():
            cmd = [
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
            run(cmd, logs / f"gt_sel{sel_tag}.log", check=True)

        # Search sweep
        cmd = [
            "numactl", "--cpunodebind=1", "--membind=1",
            str(search_bin),
            "--data_type", "uint8",
            "--dist_fn", "l2",
            "--index_path_prefix", str(index_prefix),
            "--result_path", str(out / f"results/sel{sel_tag}"),
            "--query_file", str(sub_q),
            "--gt_file", str(sub_gt),
            "--filter_label", str(label),
            "--label_type", "uint",
            "-W", "4",
            "--num_nodes_to_cache", "100000",
            "-T", str(args.threads_search),
            "-K", "10",
            "-L", *[str(v) for v in COARSE_LS],
        ]
        out_text = run(cmd, logs / f"search_sel{sel_tag}.log", check=True)
        rows = parse_search_rows(out_text)
        best = None
        for r in rows:
            if r["recall"] >= args.recall_target:
                best = r
                break
        if best is None and rows:
            best = max(rows, key=lambda x: x["recall"])

        if best is not None:
            csv_rows.append(
                {
                    "selectivity_target": target,
                    "selectivity_actual": actual_sel,
                    "filter_label": label,
                    "query_count": nq,
                    "threads": args.threads_search,
                    "min_L_for_target": best["L"],
                    "recall": best["recall"],
                    "qps": best["qps"],
                    "avg_cmps": best["avg_cmps"],
                    "latency_us": best["latency_us"],
                    "p99_latency_us": best["p99_latency_us"],
                }
            )
            print(
                f"sel={target:.3f} actual={actual_sel:.4f} label={label} nq={nq} "
                f"L={best['L']} recall={best['recall']:.2f} qps={best['qps']:.2f}"
            )

    out_csv = out / "yfcc10m_filteredvamana_ssd_repro.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "selectivity_target", "selectivity_actual", "filter_label", "query_count", "threads",
                "min_L_for_target", "recall", "qps", "avg_cmps", "latency_us", "p99_latency_us",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
