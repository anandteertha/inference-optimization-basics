# Lesson 4: Batching experiments

[Lesson notes](04-batching-and-serving.md) · [Course overview](../README.md)

Compare a deterministic scheduling simulation with a measured CPU matrix operation. The simulation exposes queueing effects; the CPU experiment illustrates execution efficiency. Neither is an end-to-end language-model benchmark.

## Practical 1: Scheduling simulation

Script: [batching_lab.py](../labs/batching_lab.py). One FIFO worker, regular arrivals, hypothetical size-dependent costs, 1,000 requests, no network or model execution. Units below are milliseconds unless stated otherwise. These are **simulated** request latencies.

```powershell
python labs/batching_lab.py --batch 1 --timeout 0 --rate 200
python labs/batching_lab.py --batch 4 --timeout 0 --rate 200
python labs/batching_lab.py --batch 4 --timeout 15 --rate 200
```

| Maximum batch | Timeout | Typical actual batch | Mean latency |  p50 |  p99 | Completed requests/s including drain |
| ------------: | ------: | -------------------- | -----------: | ---: | ---: | -----------------------------------: |
|             1 |       0 | 1                    |       1506.5 | 1505 | 2975 |                                125.0 |
|             4 |       0 | 2 after startup      |         13.5 |   11 |   16 |                                199.8 |
|             4 |      15 | 4                    |         19.5 |   17 |   27 |                                199.7 |

With batch one, zero-indexed request n has latency `8 + 3n` ms: arrivals are 5 ms apart but execution takes 8 ms. Nearest-rank p99 for 1,000 requests selects n = 989, giving 2975 ms. Continued arrivals would grow the queue; this finite-run p99 is not a stationary long-run percentile.

For maximum four and zero timeout, launches occur at 0, 8, 16, and 26 ms with sizes 1, 1, 2, and 2. After startup, two requests execute for 10 ms and two more arrive during that execution. The worker stays busy in this perfectly regular repeating pattern. This does not demonstrate robustness to bursts.

For timeout 15 ms, the first full batch launches at 15 ms and finishes at 27 ms. Arrival-relative latencies are 27, 22, 17, and 12 ms. Subsequent full batches start 20 ms apart, executing for 12 ms and leaving 8 ms idle. The worker is busy 60% of the repeating cycle, versus 100% with zero timeout. This measures simulated worker occupancy, not hardware utilization or energy use.

Thus, waiting improves amortized execution cost from 5 to 3 ms per item, but raises simulated p99 from 16 to 27 ms while both policies satisfy the supplied rate of approximately 200 requests/s. Both meet the 50 ms target in this simulated run. Greater execution capacity need not increase achieved throughput when demand is fixed.

## Practical 2: CPU matrix multiplication

Script: [batching_cpu.py](../labs/batching_cpu.py). Command from the repository root:

```powershell
python labs/batching_cpu.py
```

Reported configuration: Windows, PyTorch 2.10.0+cpu, one CPU thread, float32, `[B × 4096] @ [4096 × 4096]`, fixed 64.0 MiB weight matrix. The script defaults to seven repeats and 100 ms target blocks after warmup/calibration. The saved output does not include repeat counts, the CPU model, the run timestamp, or raw per-repeat samples; treat this table as an illustrative recorded run rather than a fully specified benchmark. All output checks passed against the float64 reference.

|    B | batch_ms | item_ms | capacity_items/s | speedup_vs_Bx1 | max_abs_error |
| ---: | -------: | ------: | ---------------: | -------------: | ------------: |
|    1 |   4.6261 |  4.6261 |            216.2 |           1.00 |      5.60e-06 |
|    2 |  11.4596 |  5.7298 |            174.5 |           0.81 |      9.54e-07 |
|    4 |  12.0120 |  3.0030 |            333.0 |           1.54 |      1.31e-06 |
|    8 |  18.7331 |  2.3416 |            427.1 |           1.98 |      1.43e-06 |
|   16 |  24.9288 |  1.5580 |            641.8 |           2.97 |      1.43e-06 |
|   32 |  48.9209 |  1.5288 |            654.1 |           3.03 |      1.91e-06 |

Interpretation:

- Batch two is slower than two separate batch-one executions: 11.4596 ms versus approximately 9.2522 ms. The cause is unresolved; timings alone cannot identify a library strategy change or variability.
- Batch sixteen processes sixteen items in 24.9288 ms versus approximately 74.0176 ms for sixteen individual executions, a 2.97× capacity improvement.
- Moving from sixteen to thirty-two raises capacity by approximately 1.9%, while batch time rises approximately 96.2%.
- For a purely illustrative 50 ms budget using these medians as fixed costs, batch thirty-two leaves 1.0791 ms for all other work; batch sixteen leaves 25.0712 ms. This makes sixteen a candidate to investigate, not a proven serving policy.

`item_ms = batch_ms/B`; `capacity_items/s = 1000B/batch_ms`; `speedup_vs_Bx1 = B × batch_one_ms/batch_ms`. Item cost is amortized. Every item participates in the full batch execution.

These are medians of execution block averages. They exclude assembly, queueing, networking, input creation, allocation, token generation, and attention. They do not establish serving p99. One thread may still use vector instructions. Data reuse and library strategy can contribute, but no memory-traffic profile isolates their contributions. This is a repeated-weight CPU operation, not a GPU or end-to-end LLM benchmark.
