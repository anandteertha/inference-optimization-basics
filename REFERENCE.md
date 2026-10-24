# Lessons 1-3: Quick reference

[Course overview](README.md) · [Course status](COURSE_STATUS.md)

## Timing

For N output tokens with constant inter-token interval d and request start at time zero:

```text
Final-token time = TTFT + (N - 1) × d
Subsequent-token rate = 1 / d
Whole-request output rate = N / final-token time
```

For variable intervals, add the actual gaps. For N > 1, mean ITL = (last arrival - first arrival) / (N - 1). Its reciprocal describes the aggregate rate across that interval, not the mean of individual instantaneous rates. With one token, no ITL is observed.

Specify timing boundaries and token-count conventions. One request's rate is distinct from a server's total throughput.

## Generation lifecycle

```text
Process prompt → next-token scores → select first output token
Process that selected token → new scores → select following token
Repeat until a stopping condition
```

Selecting a token does not yet add that token's own keys and values to the cache. Processing it as input does.

For the hypothetical six-token France prompt:

| Event | Cached positions per attention layer |
| --- | ---: |
| Paris selected after prefill | 6 |
| Paris processed; period selected | 7 |
| Period processed; EOS selected | 8 |

With 34 cached attention layers, each has K and V entries for the same positions. This does not multiply the sequence length by 34.

## Two different forms of reuse

| Reuse | What it saves |
| --- | --- |
| Previous K/V entries | Recalculating those entries; attention still accesses them |
| Loaded weight pieces across input rows | Repeated weight movement; input-specific arithmetic remains |

For a given layer, Q = X W_Q, K = X W_K, V = X W_V. The matrices are shared across positions and fixed during ordinary inference. Layer inputs and generally the matrices differ across layers.

Prefill has many known input rows available together. A single-request decode step has one new row. This changes reuse opportunities; it does not change the learned matrices.

## Resources and bottlenecks

| Resource | Question |
| --- | --- |
| Capacity | Does the data fit? |
| Bandwidth | How fast can the data move? |
| Compute throughput | How fast can arithmetic run? |

Under ideal complete overlap, unchanged work, and no other overhead:

```text
Step time = max(arithmetic time, transfer time)
```

Real execution may take longer. Faster arithmetic does not directly shorten transfer time; faster transfer does not directly shorten arithmetic time.

| Hypothetical change | Compute | Transfer | Ideal step time |
| --- | ---: | ---: | ---: |
| Baseline | 2 ms | 8 ms | 8 ms |
| Double compute speed | 1 ms | 8 ms | 8 ms |
| Double bandwidth instead | 2 ms | 4 ms | 4 ms |

**Rule:** Improve the resource that limits the actual workload. A useful optimization may leave the same resource as the bottleneck.

## Read more slowly when needed

- [Lesson 1: Definitions and worked timelines](lessons/01-inference-metrics.md)
- [Lesson 2: The full prompt-to-answer walkthrough](lessons/02-prefill-decode-kv-cache.md)
- [Lesson 3: Weight reuse and resource reasoning](lessons/03-compute-memory-bottlenecks.md)
