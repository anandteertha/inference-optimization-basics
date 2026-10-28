# inference-optimization-basics

A foundations-first course in understanding, measuring, and optimizing language-model inference.

This repository focuses on autoregressive transformer language models: how they process prompts, generate tokens, use GPU resources, and respond to optimization. The goal is to make performance decisions using an understanding of the workload and evidence from measurements.

## Learning outcomes

By the end of this course, you should be able to:

- Distinguish time to first token, inter-token latency, total response time, and throughput.
- Explain prefill, decode, attention, and KV-cache reuse.
- Reason about compute, memory bandwidth, and memory capacity as separate constraints.
- Explain tradeoffs behind batching, quantization, attention optimizations, caching, and speculative decoding.
- Establish a reproducible inference baseline and compare optimizations fairly.
- Choose an optimization for specific latency, throughput, memory, and output-quality requirements.

## Learning approach

The course has two phases: eight conceptual lessons, followed by four practical projects.

1. Complete the conceptual lessons in order, one lesson at a time.
2. Use short calculations and conceptual exercises to check understanding after each lesson.
3. Review answers and resolve misunderstandings before advancing.
4. Begin coding after completing the conceptual phase.
5. Apply the theory in practical projects, revisiting relevant concepts as needed.

Written lesson exercises are practice, not separate projects. Each practical project is also its corresponding lab: **Project 1 = Lab 1**, and so on.

A suggested daily session is 45-60 minutes: brief recall, one focused lesson, an exercise, and review. Two short lessons may fit in a session; a difficult lesson or coding lab may need several sessions. Understanding determines progression rather than a daily lesson quota.

## Read the completed lessons

Lessons 1-4 are available. Start with the notes below for explanations, worked examples, diagrams, and self-check exercises with answers.

| Reading | Contents |
| --- | --- |
| [Lesson 1: Inference metrics](lessons/01-inference-metrics.md) | Token timelines, timing formulas, and the difference between two token-rate definitions |
| [Lesson 2: Prefill, decode, and KV cache](lessons/02-prefill-decode-kv-cache.md) | The six-token France prompt, first-token selection, and the cache at every stage |
| [Lesson 3: Compute and memory bottlenecks](lessons/03-compute-memory-bottlenecks.md) | Projection matrices, weight reuse, resource limits, overlap, and worked calculations |
| [Lesson 4: Batching and serving](lessons/04-batching-and-serving.md) | Weight reuse across requests, scheduling traces, queueing, tail latency, and memory constraints |
| [Lesson 4: Batching experiments](lessons/04-batching-experiments.md) | Runnable scheduling simulation, CPU matrix-multiplication example, and interpretation of results |
| [Quick reference](REFERENCE.md) | Formulas and distinctions for a short refresher |
| [Course status](COURSE_STATUS.md) | Available material and upcoming topics |

Recommended catch-up order: read the quick reference, revisit any unclear lesson, then try its recall questions before opening the answers. Lesson 5 is the next planned addition. Coding begins after Lesson 8; Lesson 4 also links optional demonstrations already available in the repository.

## Prerequisites

- Basic programming knowledge; familiarity with Python is useful for the practical phase.
- Basic arithmetic and comfort with units such as milliseconds, bytes, and tokens per second.
- A willingness to state assumptions and explain results in your own words.

The conceptual phase requires no GPU or paid API. Hardware and model choices for measured labs will be made before those labs begin. Supported optimizations depend on the model, hardware, and inference engine selected.

## Phase 1: Conceptual foundations

| Lesson | Topics | Understanding checkpoint |
| --- | --- | --- |
| **1. Inference metrics** | Time to first token (TTFT), inter-token latency (ITL), total response time, and throughput | Calculate response timings and distinguish subsequent-token generation rate from whole-request output throughput. |
| **2. Prefill, decode, and KV cache** | Prompt processing, parallelism across prompt positions, sequential generation, and cached attention keys and values | Explain what the cache avoids recomputing, what still needs to be read, and why context length matters. |
| **3. Compute and memory bottlenecks** | Arithmetic throughput, memory bandwidth versus capacity, weight reuse, and a simplified timing model | Identify the limiting resource and explain why accelerating another resource may have little effect. |
| **4. Batching and serving** | Static and continuous batching, concurrency, queueing, tail latency, and latency-throughput tradeoffs | Explain how batching improves reuse and why higher throughput need not mean lower request latency. |
| **5. Quantization and memory use** | Precision, weight-memory estimates, weight versus activation/KV-cache quantization, and quality-performance tradeoffs | Estimate memory savings and explain why lower precision does not guarantee proportional speedups. |
| **6. Attention and KV-cache optimizations** | Attention data movement, FlashAttention intuition, cache sizing, multi-head/grouped-query/multi-query attention, paged KV storage, and prefix caching | Distinguish changes to attention computation, cache size, cache allocation, and reuse across requests. |
| **7. Speculative decoding** | Draft generation, target-model verification, acceptance rate, correctness conditions, and overhead | Explain when speculation can help and when overhead can outweigh its benefit. |
| **8. Benchmarking and optimization decisions** | Representative workloads, warmup, timing boundaries, latency distributions, throughput, memory, quality checks, and controlled comparisons | Design a comparison that isolates a change and supports a workload-specific conclusion. |

The conceptual phase is complete when the checkpoints have been reviewed and remaining misunderstandings resolved. Reading or presenting a lesson alone does not establish completion.

## Phase 2: Practical projects and labs

### Project 1 / Lab 1: Token timing analyzer

Build a Python tool that analyzes output-token arrival timestamps.

- Calculate TTFT, successive token intervals, average ITL, and total response time.
- Distinguish subsequent-token generation rate from whole-request output throughput.
- Handle insufficient or invalid input explicitly.
- Check calculations against small examples with known answers.

**Deliverable:** a working analyzer, example inputs and outputs, and an explanation of each metric. Clearly label synthetic data; this project alone is not a model-performance benchmark.

### Project 2 / Lab 2: Establish a measured baseline

Run an accessible model and measure a defined workload before changing optimization settings.

- Record hardware, model, precision, inference engine, and relevant versions.
- Define prompt lengths, output lengths, concurrency, and timing boundaries.
- Separate warmup from measured runs and repeat measurements.
- Report latency, throughput, and memory using clearly stated definitions.

**Deliverable:** a reproducible baseline with recorded configuration, measurements, and limitations.

### Project 3 / Lab 3: Controlled optimization experiments

Test supported changes such as batch size, context length, precision, or cache settings.

- Change one factor at a time and retain a comparable baseline.
- Predict the effect using bottleneck concepts before measuring it.
- Compare latency, throughput, memory, and output quality as applicable.
- Explain unexpected results and identify tradeoffs.

**Deliverable:** an experiment report with configurations, results, and evidence-backed conclusions. Not every conceptual technique must be implemented in this basics course.

### Project 4 / Lab 4: Optimization capstone

Choose a workload and define latency, throughput, memory, and quality requirements. Identify a bottleneck, select a supported optimization, and evaluate it against the baseline.

**Deliverable:** reproducible before-and-after results, a justified recommendation, and a clear account of limitations and unsuccessful experiments.

## Standards for practical work

- Keep measured results separate from hypothetical examples and simulations.
- Include units, configurations, and workload details with every performance claim.
- Use consistent measurement boundaries and account for asynchronous execution where relevant.
- Check output quality when changing numerical precision or generation behavior.
- Explain which workload benefits; avoid treating one result as a universal speedup.

The roadmap is fully contained in this README. Project files will be added as the practical phase begins; no additional files are required to understand or start the conceptual phase.

## Beyond basics

Possible follow-on repositories are `inference-optimization-intermediate` and `inference-optimization-advanced`. Their scope will be defined after this course. Potential topics include deeper GPU profiling, kernel fusion, compilation and CUDA graphs, tensor/pipeline parallelism, and prefill/decode disaggregation.
