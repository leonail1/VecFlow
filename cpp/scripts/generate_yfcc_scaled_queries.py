#!/usr/bin/env python3
import argparse
import random
import struct
from pathlib import Path


def read_u8bin(path: Path) -> tuple[int, int, bytes]:
    with path.open('rb') as f:
        rows, dim = struct.unpack('<II', f.read(8))
        payload = f.read()
    expected = rows * dim
    if len(payload) != expected:
        raise RuntimeError(f'{path} payload size mismatch: got {len(payload)}, expected {expected}')
    return rows, dim, payload


def read_spmat(path: Path) -> tuple[int, int, list[int], list[int]]:
    with path.open('rb') as f:
        nrow, ncol, nnz = struct.unpack('<qqq', f.read(24))
        indptr = list(struct.unpack(f'<{nrow + 1}q', f.read(8 * (nrow + 1))))
        indices = list(struct.unpack(f'<{nnz}i', f.read(4 * nnz)))
    return nrow, ncol, indptr, indices


def write_u8bin(path: Path, rows: int, dim: int, payload: bytearray) -> None:
    with path.open('wb') as f:
        f.write(struct.pack('<II', rows, dim))
        f.write(payload)


def write_spmat(path: Path, nrow: int, ncol: int, indptr: list[int], indices: list[int]) -> None:
    with path.open('wb') as f:
        f.write(struct.pack('<qqq', nrow, ncol, len(indices)))
        f.write(struct.pack(f'<{len(indptr)}q', *indptr))
        if indices:
            f.write(struct.pack(f'<{len(indices)}i', *indices))


def main() -> int:
    parser = argparse.ArgumentParser(description='Scale YFCC single-label queries by random resampling')
    parser.add_argument('--base-query', required=True)
    parser.add_argument('--base-label', required=True)
    parser.add_argument('--output-query', required=True)
    parser.add_argument('--output-label', required=True)
    parser.add_argument('--target-count', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20260417)
    args = parser.parse_args()

    base_query = Path(args.base_query)
    base_label = Path(args.base_label)
    output_query = Path(args.output_query)
    output_label = Path(args.output_label)

    rows, dim, payload = read_u8bin(base_query)
    nrow, ncol, indptr, indices = read_spmat(base_label)
    if nrow != rows:
        raise RuntimeError(f'query rows mismatch: {rows} vs label rows {nrow}')

    rng = random.Random(args.seed)
    sample_ids = [rng.randrange(rows) for _ in range(args.target_count)]

    out_payload = bytearray(args.target_count * dim)
    out_indptr = [0]
    out_indices: list[int] = []
    for out_row, src_row in enumerate(sample_ids):
        src_start = src_row * dim
        src_end = src_start + dim
        dst_start = out_row * dim
        out_payload[dst_start:dst_start + dim] = payload[src_start:src_end]

        label_start = indptr[src_row]
        label_end = indptr[src_row + 1]
        out_indices.extend(indices[label_start:label_end])
        out_indptr.append(len(out_indices))

    output_query.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    write_u8bin(output_query, args.target_count, dim, out_payload)
    write_spmat(output_label, args.target_count, ncol, out_indptr, out_indices)
    print(f'wrote {output_query}')
    print(f'wrote {output_label}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
