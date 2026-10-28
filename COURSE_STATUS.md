# Course status

Updated: October 28, 2026.

This page tracks the material available in the repository.

## Available lessons

| Lesson | Material |
| --- | --- |
| [1. Inference metrics](lessons/01-inference-metrics.md) | Definitions, token timelines, worked calculations, and self-check questions |
| [2. Prefill, decode, and KV cache](lessons/02-prefill-decode-kv-cache.md) | A prompt-to-answer walkthrough, per-layer cache structure, and self-check questions |
| [3. Compute and memory bottlenecks](lessons/03-compute-memory-bottlenecks.md) | Resource definitions, projection matrices, weight reuse, overlap, and optimization comparisons |
| [4. Batching and serving](lessons/04-batching-and-serving.md) | Static, dynamic, and continuous batching, worked traces, queueing, tail latency, KV capacity, and recall questions |

A [quick reference](REFERENCE.md) summarizes these lessons.

## Planned lessons

5. Quantization and memory use.
6. Attention and KV-cache optimizations.
7. Speculative decoding.
8. Benchmarking and optimization decisions.

## Planned practical projects

Practical projects follow the conceptual lessons. Each project is its corresponding lab.

1. Token timing analyzer.
2. Measured model baseline.
3. Controlled optimization experiments.
4. Optimization capstone.

The [README](README.md) contains the full roadmap, prerequisites, expected deliverables, and suggested learning approach.

## Lesson 4 experiments

The [batching experiments](lessons/04-batching-experiments.md) include a standard-library scheduling simulator and a PyTorch CPU matrix-multiplication example. Results distinguish simulated request latencies from measured execution times and explain the limits of each comparison.
