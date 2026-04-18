#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert a CSR .spmat label file to DiskANN txt labels')
    parser.add_argument('--input', required=True, help='Input .spmat path')
    parser.add_argument('--output', required=True, help='Output .txt path')
    parser.add_argument('--empty-token', default='-1', help='Token for rows without labels')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open('rb') as f:
        nrow, ncol, nnz = struct.unpack('qqq', f.read(24))
        indptr = struct.unpack(f'{nrow + 1}q', f.read(8 * (nrow + 1)))
        indices = struct.unpack(f'{nnz}i', f.read(4 * nnz))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as out:
        for row in range(nrow):
            start = indptr[row]
            end = indptr[row + 1]
            if start == end:
                out.write(args.empty_token + '\n')
            else:
                out.write(','.join(str(indices[i]) for i in range(start, end)) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
