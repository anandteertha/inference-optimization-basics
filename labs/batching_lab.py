"""Lesson 4: deterministic batching simulation. Python standard library only.

Times are simulated milliseconds, NOT laptop/model measurements.
One FIFO worker; oldest-request timeout runs even while worker is busy.
Arrivals at a launch timestamp are eligible for that batch.
No network, preprocessing, padding, failures or concurrent workers.
Execution costs below are hypothetical and depend on actual batch size.

Run: python batching_lab.py --batch 1 --timeout 0 --rate 200
Then: python batching_lab.py --batch 4 --timeout 15 --rate 200
Inspect the first trace before comparing aggregate metrics.
Next experiments: rates 10, 100, 200; timeouts 0, 5, 15, 50.
Questions: Which policies meet p99 <= 50 ms? Does a shorter timeout
ever REDUCE throughput and increase p99? Explain using actual batch sizes.
Finite runs drain all requests: always inspect final-window latency too.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class Result:
    request: int
    arrival: float
    start: float
    finish: float
    size: int

    @property
    def latency(self):
        return self.finish - self.arrival


def simulate(arrivals, max_batch, timeout, costs):
    """Event-based execution; no sleeping and no wall-clock benchmarking."""
    if max_batch not in costs or timeout < 0:
        raise ValueError('Unsupported batch size or negative timeout')
    if any(not math.isfinite(a) or a < 0 for a in arrivals):
        raise ValueError('Arrivals must be finite nonnegative milliseconds')
    if any(a > b for a, b in zip(arrivals, arrivals[1:])):
        raise ValueError('Arrivals must be sorted')
    rows = []
    first = 0
    free_at = 0.0
    while first < len(arrivals):
        # An idle worker waits for the first request. A busy worker cannot launch.
        available = max(free_at, arrivals[first])
        deadline = arrivals[first] + timeout
        # When will max_batch requests be available? Infinity if stream ends.
        full_index = first + max_batch - 1
        full_at = arrivals[full_index] if full_index < len(arrivals) else math.inf
        start = max(available, min(deadline, full_at))
        end = first
        while end < len(arrivals) and end - first < max_batch and arrivals[end] <= start:
            end += 1
        size = end - first
        finish = start + costs[size]
        rows.extend(Result(i, arrivals[i], start, finish, size) for i in range(first, end))
        free_at = finish
        first = end
    return rows


def percentile(values, p):
    """Nearest-rank percentile; intentionally explicit for reproducibility."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--timeout', type=float, default=0)
    parser.add_argument('--rate', type=float, default=200)
    parser.add_argument('--requests', type=int, default=1000)
    parser.add_argument('--csv', help='Optional per-request CSV output path')
    args = parser.parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0 or args.requests < 1:
        parser.error('rate must be finite and positive; requests must be positive')
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error('timeout must be finite and nonnegative')
    # Explicit illustrative costs for ALL possible actual batch sizes.
    costs = {1: 8, 2: 10, 3: 11, 4: 12, 5: 14, 6: 16, 7: 18, 8: 20}
    arrivals = [i * 1000 / args.rate for i in range(args.requests)]
    rows = simulate(arrivals, args.batch, args.timeout, costs)
    print('SIMULATED results; hypothetical execution costs; regular arrivals')
    print('id  arrival  start  finish  batch  wait  latency (ms)')
    for r in rows[:12]:
        print(f'{r.request:2} {r.arrival:8.1f} {r.start:6.1f} {r.finish:7.1f}'
              f' {r.size:5} {r.start-r.arrival:6.1f} {r.latency:7.1f}')
    latency = [r.latency for r in rows]
    duration = rows[-1].finish - arrivals[0]
    print(f'Offered rate: {args.rate:.1f} requests/s')
    print(f'Completed / elapsed time including drain: {len(rows)*1000/duration:.1f} requests/s')
    print(f'Mean latency: {mean(latency):.1f} ms')
    print(f'p50: {percentile(latency, .5):.1f} ms; p99: {percentile(latency, .99):.1f} ms')
    print(f'Last 10% mean latency: {mean(latency[-max(1,len(rows)//10):]):.1f} ms')
    print(f'Fraction within 50 ms: {sum(x <= 50 for x in latency)/len(rows):.1%}')
    if args.csv:
        with open(args.csv, 'w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['request', 'arrival_ms', 'start_ms', 'finish_ms',
                             'batch_size', 'waiting_ms', 'latency_ms'])
            for r in rows:
                writer.writerow([r.request,r.arrival,r.start,r.finish,r.size,
                                 r.start-r.arrival,r.latency])


if __name__ == '__main__':
    main()
