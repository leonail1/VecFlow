#!/usr/bin/env python3
"""FilteredVamana memory-index recall reproduction with single filter_label.

Goal: emulate node6 evaluation style on current environment without SSD build instability.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import subprocess
from pathlib import Path

COARSE_LS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000]


def run(cmd: list[str], log: Path | None = None, check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(out)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{out[-2000:]}")
    return out


def parse_search_rows(output: str):
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


def write_u8bin(path: Path, n: int, d: int, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("II", n, d))
        f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/lzg/VecFlow-develop")
    ap.add_argument("--threads-build", type=int, default=52)
    ap.add_argument("--threads-search", type=int, default=1)
    ap.add_argument("--recall-target", type=float, default=98.0)
    ap.add_argument("--nq", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.root)
    ds = root / "vecflow/datasets/yfcc10M"
    out = root / "experiment_results/three_way_comparison/filteredvamana_memory_repro"
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)

    mapping = root / "experiment_results/three_way_comparison/filteredvamana_memory_repro_labels.tsv"
    build_bin = root / "thirdparty/DiskANN/build/apps/build_memory_index"
    search_bin = root / "thirdparty/DiskANN/build/apps/search_memory_index"
    gt_bin = root / "thirdparty/DiskANN/build/apps/utils/compute_groundtruth_for_filters"

    base_file = ds / "base.10M.u8bin"
    query_file = ds / "query.public.single_label.u8bin"
    base_labels = ds / "base_labels_diskann.txt"

    index_prefix = out / "indices/yfcc10m_filteredvamana_mem_R64_L100_F200"
    idx_file = Path(str(index_prefix))
    if not idx_file.exists():
        index_prefix.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "numactl", "--cpunodebind=1", "--membind=1",
            str(build_bin),
            "--data_type", "uint8",
            "--dist_fn", "l2",
            "--data_path", str(base_file),
            "--index_path_prefix", str(index_prefix),
            "-R", "64",
            "-L", "100",
            "--FilteredLbuild", "200",
            "--alpha", "1.2",
            "-T", str(args.threads_build),
            "--label_file", str(base_labels),
            "--label_type", "uint",
        ]
        run(cmd, logs / "build.log", check=True)

    nq_all, dim, qdata = read_u8bin(query_file)
    rows_out = []
    with mapping.open() as f:
        next(f)
        for line in f:
            target_s, label_s, actual_s, base_count_s, qcount_s = line.strip().split("\t")
            target = float(target_s)
            label = int(label_s)

            # find query ids with this label
            qids = []
            with (ds / "query_filters_diskann.txt").open() as qf:
                for i, qline in enumerate(qf):
                    s = qline.strip()
                    if not s or s == "-1":
                        continue
                    if int(s.split(",")[0]) == label:
                        qids.append(i)
            qids = qids[: args.nq]
            if not qids:
                continue

            # subset query file and filter file
            sub = bytearray()
            for qid in qids:
                st = qid * dim
                sub.extend(qdata[st: st + dim])

            tag = str(target).replace(".", "p")
            sub_q = out / f"cache/query_sel{tag}_n{len(qids)}.u8bin"
            sub_f = out / f"cache/query_sel{tag}_n{len(qids)}.filters.txt"
            sub_gt = out / f"cache/gt_sel{tag}_n{len(qids)}.bin"
            write_u8bin(sub_q, len(qids), dim, bytes(sub))
            sub_f.write_text("\n".join([str(label)] * len(qids)) + "\n")

            # gt
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
                run(cmd, logs / f"gt_sel{tag}.log", check=True)

            # search sweep
            cmd = [
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
            out_text = run(cmd, logs / f"search_sel{tag}.log", check=True)
            metrics = parse_search_rows(out_text)
            best = None
            for m in metrics:
                if m["recall"] >= args.recall_target:
                    best = m
                    break
            if best is None and metrics:
                best = max(metrics, key=lambda x: x["recall"])
            if not best:
                continue

            rows_out.append({
                "selectivity_target": target,
                "label": label,
                "query_count": len(qids),
                "threads": args.threads_search,
                "best_L": best["L"],
                "recall": best["recall"],
                "qps": best["qps"],
                "latency_us": best["latency_us"],
                "p99_latency_us": best["p99_latency_us"],
            })
            print(f"sel={target:.3f} label={label} nq={len(qids)} L={best['L']} recall={best['recall']:.2f} qps={best['qps']:.2f}")

    out_csv = out / "yfcc10m_filteredvamana_memory_repro.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "selectivity_target", "label", "query_count", "threads", "best_L", "recall", "qps", "latency_us", "p99_latency_us"
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
