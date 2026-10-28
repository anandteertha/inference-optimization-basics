"""Lesson 4: measure CPU float32 matrix multiplication versus batch size.

Run: python batching_cpu.py
One thread by default. No model downloads or GPU needed.
Measures preallocated torch.mm output; excludes allocation, input creation,
networking, queueing, and batch assembly. This is a repeated-weight workload.
Medians summarize block-average execution times, NOT request-latency p99.
Small enough weights may stay in CPU cache: do not infer DRAM traffic.
"""

import argparse
import csv
import platform
import random
import statistics
from time import perf_counter

import torch


def time_block(x, weights, output, loops):
    start = perf_counter()
    for _ in range(loops):
        torch.mm(x, weights, out=output)
    return (perf_counter() - start) / loops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--dim', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--block-ms', type=float, default=100)
    parser.add_argument('--csv', help='Optional summary output file')
    args = parser.parse_args()
    if min(args.threads, args.dim, args.repeats) < 1 or not (1 <= args.block_ms <= 10000):
        parser.error('positive threads/dim/repeats required; block-ms must be 1..10000')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(4)
    sizes = [1, 2, 4, 8, 16, 32]
    print(f'PyTorch {torch.__version__}; {platform.system()}; threads={torch.get_num_threads()}')
    print(f'CPU float32; K=N={args.dim}; weights={args.dim**2*4/2**20:.1f} MiB')
    print('Times are measured execution block averages; not serving latency or p99.')
    print('Checking correctness, warming up, and calibrating each batch size...', flush=True)
    with torch.inference_mode():
        weights = torch.randn(args.dim, args.dim) / args.dim**0.5
        inputs = torch.randn(max(sizes), args.dim)
        # Mathematical oracle: float64 products for the same float32 inputs.
        oracle = inputs.double() @ weights.double()
        cases = {}
        for b in sizes:
            x = inputs[:b].contiguous()
            output = torch.empty(b, args.dim)
            torch.mm(x, weights, out=output)
            reference = oracle[:b].float()
            torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-4)
            max_error = (output-reference).abs().max().item()
            for _ in range(10):
                torch.mm(x, weights, out=output)
            loops = 1
            while True:
                elapsed = time_block(x, weights, output, loops)
                if elapsed * loops >= args.block_ms / 1000:
                    break
                loops *= 2
            cases[b] = (x, output, loops, max_error)
        samples = {b: [] for b in sizes}
        rng = random.Random(4)
        for _ in range(args.repeats):
            order = sizes.copy()
            rng.shuffle(order)
            for b in order:
                x, output, loops, _ = cases[b]
                samples[b].append(time_block(x, weights, output, loops))
    baseline = statistics.median(samples[1])
    rows = []
    print('All output checks passed against float64 reference.')
    print(' B   batch_ms   item_ms   capacity_items/s   speedup_vs_Bx1   max_abs_error')
    for b in sizes:
        seconds = statistics.median(samples[b])
        row = dict(batch=b, batch_ms=seconds*1000, item_ms=seconds*1000/b,
                   capacity_items_s=b/seconds, speedup_vs_Bx1=b*baseline/seconds,
                   max_abs_error=cases[b][3],
                   min_block_ms=min(samples[b])*1000,
                   max_block_ms=max(samples[b])*1000)
        rows.append(row)
        print(f'{b:2} {row["batch_ms"]:10.4f} {row["item_ms"]:9.4f}'
              f' {row["capacity_items_s"]:18.1f} {row["speedup_vs_Bx1"]:16.2f}'
              f' {row["max_abs_error"]:15.2e}')
    if args.csv:
        with open(args.csv, 'w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
