#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - << 'PY'
import json, csv, pathlib
root = pathlib.Path('.')
# check json parse
for p in [
    root/'results'/'develop_bench_rebuild.json',
    root/'results'/'original_vecflow_results.json',
    root/'results'/'develop_timeseries.json',
]:
    with open(p) as f:
        json.load(f)
# check csv parse
with open(root/'results'/'yfcc10m_filteredvamana_reuse_index_repro.csv') as f:
    rows=list(csv.DictReader(f))
assert len(rows)>0
print('json/csv parse OK')
PY

python3 scripts/plot_vecflow_qps_timeseries.py \
  --series develop=results/develop_timeseries.json \
  --bucket-seconds 0.5 \
  --output-dir results/quick_check_plots

echo "quick check done: results/quick_check_plots"
