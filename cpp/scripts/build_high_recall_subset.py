#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path


def read_u8bin(path: Path):
    with path.open('rb') as f:
        n, d = struct.unpack('II', f.read(8))
        data = f.read(n * d)
    return n, d, data


def write_u8bin(path: Path, n: int, d: int, rows: bytes):
    with path.open('wb') as f:
        f.write(struct.pack('II', n, d))
        f.write(rows)


def read_spmat(path: Path):
    with path.open('rb') as f:
        nrow, ncol, nnz = struct.unpack('qqq', f.read(24))
        indptr = list(struct.unpack(f'{nrow+1}q', f.read(8 * (nrow + 1))))
        indices = list(struct.unpack(f'{nnz}i', f.read(4 * nnz)))
    return nrow, ncol, indptr, indices


def write_spmat(path: Path, nrow: int, ncol: int, indptr, indices):
    nnz = len(indices)
    with path.open('wb') as f:
        f.write(struct.pack('qqq', nrow, ncol, nnz))
        f.write(struct.pack(f'{nrow+1}q', *indptr))
        f.write(struct.pack(f'{nnz}i', *indices))


def main():
    ap = argparse.ArgumentParser(description='Build dense-label high-recall query subset for YFCC')
    ap.add_argument('--base-labels-txt', required=True)
    ap.add_argument('--query-filters-txt', required=True)
    ap.add_argument('--query-u8bin', required=True)
    ap.add_argument('--query-spmat', required=True)
    ap.add_argument('--min-candidates', type=int, default=10)
    ap.add_argument('--max-queries', type=int, default=15000)
    ap.add_argument('--out-prefix', required=True)
    args = ap.parse_args()

    base_counts = {}
    with Path(args.base_labels_txt).open() as f:
        for line in f:
            s = line.strip()
            if not s or s == '-1':
                continue
            # single or multi labels
            for tok in s.split(','):
                if not tok:
                    continue
                lid = int(tok)
                base_counts[lid] = base_counts.get(lid, 0) + 1

    keep_ids = []
    keep_labels = []
    with Path(args.query_filters_txt).open() as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s or s == '-1':
                continue
            lid = int(s.split(',')[0])
            if base_counts.get(lid, 0) >= args.min_candidates:
                keep_ids.append(i)
                keep_labels.append(lid)
                if len(keep_ids) >= args.max_queries:
                    break

    n, d, qbytes = read_u8bin(Path(args.query_u8bin))
    row_size = d
    out_rows = bytearray()
    for qid in keep_ids:
        start = qid * row_size
        out_rows.extend(qbytes[start:start + row_size])

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_u8 = out_prefix.with_suffix('.u8bin')
    write_u8bin(out_u8, len(keep_ids), d, bytes(out_rows))

    # subset spmat
    qn, qcol, qindptr, qindices = read_spmat(Path(args.query_spmat))
    assert qn == n
    new_indptr = [0]
    new_indices = []
    for qid in keep_ids:
        s = qindptr[qid]
        e = qindptr[qid + 1]
        new_indices.extend(qindices[s:e])
        new_indptr.append(len(new_indices))
    out_sp = out_prefix.with_suffix('.spmat')
    write_spmat(out_sp, len(keep_ids), qcol, new_indptr, new_indices)

    out_ids = out_prefix.with_suffix('.ids.txt')
    out_ids.write_text('\n'.join(str(i) for i in keep_ids) + '\n')

    out_filters = out_prefix.with_suffix('.filters.txt')
    out_filters.write_text('\n'.join(str(x) for x in keep_labels) + '\n')

    print(f'kept_queries={len(keep_ids)} min_candidates={args.min_candidates}')
    print(f'u8bin={out_u8}')
    print(f'spmat={out_sp}')
    print(f'ids={out_ids}')
    print(f'filters={out_filters}')


if __name__ == '__main__':
    main()
