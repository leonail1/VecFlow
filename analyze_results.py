import json, glob, os, math

base_dir = "experiment_results/repro_bundle_20260418/results/yfcc_strict_timeseries_20260418/"
summary_file = os.path.join(base_dir, "summary_full_matrix.csv")
raw_dir = os.path.join(base_dir, "raw")

def get_stats(samples):
    if not samples: return 0.0, 0.0, 0.0
    n = len(samples)
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / (n if n > 1 else 1)
    std = math.sqrt(var)
    cv = (std / mean * 100) if mean != 0 else 0
    return mean, std, cv

print("### 1) Metrics & 2) QPS Change (%)")
summary_data = {}
try:
    with open(summary_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        header = lines[0].split(',')
        cols = {h: i for i, h in enumerate(header)}
        print(f"{'mode':<12} {'scale':<10} {'qps':<10} {'recall':<10} {'search_s':<10} {'strict_s':<10}")
        for line in lines[1:]:
            parts = line.split(',')
            m, s = parts[cols['mode']], parts[cols['scale_queries']]
            try: q = float(parts[cols['qps']])
            except: q = 0.0
            r, ss, ns = parts[cols['recall']], parts[cols['search_seconds']], parts[cols['strict_samples']]
            try: ss_f = f"{float(ss):.3f}"
            except: ss_f = ss
            print(f"{m:<12} {s:<10} {q:<10.2f} {r:<10.4} {ss_f:<10} {ns:<10}")
            if s not in summary_data: summary_data[s] = {}
            summary_data[s][m] = q

    print("\n--- QPS Change (no_dram vs full) ---")
    print(f"{'scale':<10} {'full':<10} {'no_dram':<10} {'change_%':<10}")
    scales = sorted(summary_data.keys(), key=lambda x: int(x))
    for s in scales:
        modes = summary_data[s]
        f_q, n_q = modes.get('full'), modes.get('no_dram')
        if f_q and n_q and f_q > 0:
            pct = (n_q - f_q) / f_q * 100
            print(f"{s:<10} {f_q:<10.2f} {n_q:<10.2f} {pct:<10.2f}%")
except Exception as e: print(f"Error: {e}")

print("\n### 3) run_qps_samples Statistics")
print(f"{'mode':<12} {'scale':<10} {'mean':<10} {'std':<10} {'cv%':<10}")
for fpath in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            # Handle list-based JSON or object-based
            if isinstance(data, list): data = data[0]
            samples = data.get('run_qps_samples', [])
            if samples:
                fname = os.path.basename(fpath).replace(".json", "")
                mode, scale = fname.split('_q')[0], fname.split('_q')[1].split('_')[0]
                m, st, c = get_stats(samples)
                print(f"{mode:<12} {scale:<10} {m:<10.2f} {st:<10.2f} {c:<10.2f}%")
    except: pass

print("\n### 4) Failed Modes and Error Reasons")
try:
    with open(summary_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
        header = lines[0].split(',')
        cols = {h: i for i, h in enumerate(header)}
        found = False
        for line in lines[1:]:
            parts = line.split(',')
            if parts[cols['status']] != 'ok':
                print(f"FAILED: {parts[cols['mode']]:<12} at {parts[cols['scale_queries']]:<10} | Reason: {parts[cols['error']]}")
                found = True
        if not found: print("None")
except: pass
