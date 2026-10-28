# Lesson 4: Batching and serving tradeoffs

[Previous: Compute and memory bottlenecks](03-compute-memory-bottlenecks.md) · [Course overview](../README.md) · [Quick reference](../REFERENCE.md)

All numerical serving examples use hypothetical costs; simulator outputs are simulated, not measured model performance. See the [batching experiments](04-batching-experiments.md) for runnable examples and recorded results.

**Goal:** Explain how independent requests supply work for a batch, trace static and continuous scheduling, and choose a policy using latency, throughput, and memory constraints together.

## 1. Continue the weight-reuse argument

Lesson 3 showed that one decode request supplies one new token row per iteration. Its next token is unknown until the current iteration finishes. Four independent requests, however, can each supply one ready row:

```text
X = [x_A; x_B; x_C; x_D]       shape: 4 × hidden_dimension
Q = X W_Q
```

The projection uses the same layer weights for four different inputs. Loaded weight pieces can serve multiple rows, increasing useful arithmetic per weight transfer. Each request still needs its own arithmetic, attention context, sampling state, and logical KV cache. Batching does not let A attend to B's tokens or predict A's future tokens in parallel. Specialized prefix sharing can reuse some identical context; it is not implied by batching.

For a deliberately simplified decode-step model, assume 8 ms of weight-transfer demand and 2 ms of arithmetic per request row, ideal overlap, and no other costs:

```text
T(B) = max(8, 2B) milliseconds
Aggregate output rate = 1000B / T(B) tokens/s
```

| Active requests B | Step time | Aggregate output rate | Each request's token interval |
| --- | ---: | ---: | ---: |
| 1 | 8 ms | 125 tokens/s | 8 ms |
| 4 | 8 ms | 500 tokens/s | 8 ms |
| 8 | 16 ms | 500 tokens/s | 16 ms |

Assume every request produces one counted output token per step, with no stopping or scheduling gaps. At B = 8, aggregate throughput stays high while each request gets tokens less frequently. `T(B)/B` is amortized execution cost per token, **not** an individual request's inter-token latency. A request waits for the whole step.

Real weights may be transferred more than once, attention traffic grows with contexts, and kernels and scheduling add costs. The formula explains a mechanism, not a hardware prediction. Batch growth can eventually hit compute, bandwidth, or capacity limits and can even reduce throughput.

## 2. Batch size and concurrency count different things

**Concurrency** counts requests outstanding within a stated boundary: for example, requests submitted by clients that have not completed. **Batch size** counts work grouped in a particular execution. A server can have 100 outstanding requests, 16 executing together, and 84 waiting. With continuous scheduling these counts change over time.

A maximum batch size of 16 is a limit, not proof that each batch contains 16 requests. Low arrival rates, memory limits, token budgets, and deadlines may produce smaller batches. In mixed prefill/decode execution, a token budget is also needed: one long prompt can supply many input rows while one decoding request supplies one.

## 3. Static, dynamic, and continuous batching

These terms describe different scheduling decisions; always check an engine's definitions.

**Static request batching:** choose a group and keep its membership fixed through generation. Requests may finish at different iterations. A simple implementation retains empty or masked slots until the longest request finishes. An implementation may compact finished entries, but a fixed group still cannot admit new requests until the group boundary. Completed results can be returned immediately if the implementation supports it; waiting to return them is not inherent to static membership.

**Dynamic batch formation:** accumulate compatible queued requests until a size target or waiting deadline is reached, then execute the available batch. This is common for stateless model calls. A maximum batch size and timeout govern assembly; they do not by themselves allow changing membership during autoregressive generation. NVIDIA's [dynamic batcher documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) describes this assembly mechanism.

**Continuous batching:** revisit membership at iteration boundaries. Remove completed requests and admit eligible waiting work rather than waiting for an entire group to finish. ORCA's [iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu) is a primary reference for this approach.

Consider two decode slots. A and B are already prefilled; A needs two iterations and B needs four. C is already prefilled and waiting before iteration 3, and needs two iterations. Ignore admission costs and assume equal iteration durations:

```text
Iteration                  1       2       3       4       5       6
Static fixed group       [A B]   [A B]   [- B]   [- B]   [C -]   [C -]
Continuous membership    [A B]   [A B]   [C B]   [C B]
```

After iteration 2, A leaves. Continuous scheduling gives C the available slot at iteration 3. It does not interrupt B halfway through an iteration. This trace illustrates slot utilization; it does not prove equal step times in a real engine.

An unprefilled C cannot immediately contribute a decode row. Its prompt must be processed and KV storage allocated first. Prefill may be scheduled separately, mixed with decode, or divided into chunks. Continuous batching is an admission mechanism, not a guarantee of free prefill or lower latency for every request.

## 4. Account for waiting before claiming faster service

For a single batch invocation with a fixed arrival-to-completion boundary:

```text
Request latency = waiting before execution + batch execution time
```

Waiting includes both deliberate accumulation and waiting for a busy worker. These can overlap: a batch's assembly deadline can expire while the worker is occupied. A 15 ms assembly timeout does not cap total queueing delay at 15 ms.

For an autoregressive request, TTFT includes admission/queueing, prefill, first-token selection, and delivery within the measurement boundary. After the first token, ITL can include both computation and gaps when other work is scheduled. Total response latency includes all output steps. Network and preprocessing belong in the total when the chosen boundary includes them. Avoid adding overlapping timers twice.

Here is a one-shot batching example, also used by the optional simulator. Requests arrive at 0, 5, 10, and 15 ms. A batch of one takes 8 ms; a batch of four takes 12 ms. One FIFO worker executes at a time.

| Policy | Start/finish times | Request latencies in arrival order |
| --- | --- | --- |
| Execute individually | 0–8, 8–16, 16–24, 24–32 ms | 8, 11, 14, 17 ms |
| Wait for four, timeout 15 ms | all start at 15, finish at 27 ms | 27, 22, 17, 12 ms |

The batch completes the group sooner and helps the last arrival, but delays the first arrival. Its 12/4 = 3 ms amortized cost is not any request's 3 ms latency. This simulator treats each request as one operation; it does not simulate token streaming, KV caches, or continuous LLM decoding.

## 5. Arrival rate changes the right policy

**Offered load** is work submitted per second. **Achieved throughput** is completed work per second within a stated window. **Service capacity** describes sustainable processing ability for a stated workload and policy. Keep requests/s distinct from tokens/s: output lengths can differ.

With the hypothetical costs above, an always-busy worker running individual requests has capacity `1000/8 = 125 requests/s`. Full batches of four have execution capacity `4000/12 ≈ 333.3 requests/s`, excluding assembly and other overhead. Neither figure guarantees that clients supply enough work to keep the worker busy.

At 200 requests/s, arrivals are 5 ms apart. Individual execution accumulates 3 ms more waiting per request because each operation takes 8 ms. A full batch can handle four arrivals every 20 ms in 12 ms and keep up. At 10 requests/s, arrivals are 100 ms apart: waiting 15 ms usually collects nobody else and simply adds latency.

If offered load persistently exceeds sustainable capacity, backlog grows unless work is rejected, cancelled, or redirected. A finite test eventually finishing its queue does not establish stability. Even below average capacity, bursts and variable lengths create waiting. Capacity near saturation leaves little room to absorb that variation.

For a stable system with consistent boundaries and long-run averages, Little's law relates outstanding requests L, achieved arrival/completion rate λ, and mean response time W:

```text
L = λW
Example: 100 requests/s × 0.2 s = 20 requests outstanding on average
```

L includes queued and executing requests within that boundary; it is not necessarily the batch size. Do not use an offered rate in place of the achieved stable rate during overload. This identity relates averages; it does not predict p99.

## 6. Tail latency and useful throughput

The median (p50) describes the middle of a latency distribution. The p99 describes a high percentile; record the percentile convention and sample count. The simulator uses nearest rank: sort N latencies and select rank `ceil(0.99N)`. A p99 from a short run is weak evidence about rare delays.

Define a service-level objective (SLO), such as a specified fraction of requests meeting TTFT and ITL thresholds. Track the actual passing fraction as well as percentiles. For an explicit goodput metric, count requests meeting the stated criteria per second, using the same workload and observation boundary. More total tokens/s can coexist with fewer requests meeting their latency objectives.

Long prompts, long outputs, bursts, and scheduling choices can affect the tail differently. A large prefill scheduled ahead of a decode iteration can increase ITL for an existing request. Limiting or chunking that prefill can improve responsiveness but introduces scheduling tradeoffs. No universal batch size minimizes all metrics.

## 7. Memory limits admission too

Ordinary batching shares model weights across requests, but each active request needs logical KV state. For a simple fixed-layout cache:

```text
KV bytes ≈ 2 × layers × KV_heads × head_dimension × bytes_per_element
           × sum(cached_positions_per_active_request)
```

The factor 2 accounts for keys and values. Assume a uniform cache layout and exclude allocation overhead, padding, fragmentation, and shared prefixes. Hypothetically, 32 layers, 8 KV heads, head dimension 128, and 2 bytes per element require 131,072 bytes = 128 KiB per cached position across all layers. A 4,096-position request needs 512 MiB; sixteen such requests need 8 GiB for KV entries alone.

Weights, workspaces, activations, allocator overhead, and growth during generation also need memory. A batch admitted successfully can exhaust memory later as contexts grow. Admission therefore depends on token lengths and cache policy, not just a request-count limit. Lesson 6 develops cache layouts and allocation in detail.

## 8. Optional demonstrations already in this repository

The conceptual course does not require coding before Lesson 8. These existing scripts provide optional illustrations and are not replacements for the later measured-model labs.

From the repository root, with Python available:

```powershell
python labs/batching_lab.py --batch 1 --timeout 0 --rate 200
python labs/batching_lab.py --batch 4 --timeout 15 --rate 200
python labs/batching_lab.py --batch 4 --timeout 15 --rate 10
```

Expected simulator outputs using 1,000 requests per run:

| Offered requests/s | Maximum batch | Timeout | Mean latency | p99 | Completed/s including drain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 1 | 0 ms | 1506.5 ms | 2975 ms | 125.0 |
| 200 | 4 | 15 ms | 19.5 ms | 27 ms | 199.7 |
| 10 | 4 | 15 ms | 23.0 ms | 23 ms | 10.0 |

These are deterministic simulated results from the script's cost table. The low-rate policy without waiting would have 8 ms execution latency; the timeout adds 15 ms without forming larger batches. The overloaded first run has a last-10%-of-requests mean latency of 2856.5 ms, exposing backlog growth hidden by a completion count alone. The 199.7 rate includes startup and drain, explaining why it is slightly below 200.

The optional [CPU matrix-multiplication script](../labs/batching_cpu.py) requires PyTorch. It checks results against a float64 oracle and measures repeated float32 operations across batch sizes. Its batch execution time, amortized item cost, and capacity estimates illustrate different metrics. CPU caches, kernel changes, and overhead affect results; timing alone does not prove reduced DRAM traffic. It excludes batch assembly, queueing, network, attention, and generation. Its block-average medians are not request-latency percentiles or GPU benchmark results.

## 9. Choose a serving policy from requirements

Start with prompt/output-length distributions, arrival patterns, memory limits, and latency objectives. Measure a baseline, then vary batch limits or waiting policy while keeping the workload and timing boundary consistent. Report actual batch sizes, offered and achieved rates, TTFT/ITL and total latency distributions, passing fraction, and peak memory as applicable.

For sparse interactive traffic, accumulation may buy little reuse and directly delay the first token. For queued offline work, larger batches may improve throughput within memory limits. For variable-length online generation, continuous admission can reduce vacant slots, but prefill scheduling and overload controls still matter. Rejecting work can protect admitted-request latency; report rejection rates so the result remains interpretable.

## Recall and transfer

Try these before opening the answers. The final question is a useful teach-back checkpoint.

1. How can four requests provide four decode rows without predicting four future tokens of one request?
2. Eight requests each produce a token every 20 ms. What is aggregate throughput? What is each request's ITL, and what does 20/8 represent?
3. In the two-slot trace, why can C enter iteration 3 with continuous scheduling? What prerequisite did we assume for C?
4. Why can a 15 ms assembly timeout coexist with 200 ms of queueing?
5. At 10 requests/s with arrivals exactly 100 ms apart, would a 15 ms timeout collect four requests? Explain the latency effect.
6. What is average outstanding concurrency at a stable 80 requests/s and 250 ms mean latency? Does it establish the batch size?
7. If the illustrative 4,096-position KV allocation doubles the active requests from 16 to 32, how much KV memory is needed? What memory is excluded?
8. A policy raises aggregate tokens/s but worsens p99 TTFT and ITL. Explain how that can happen and what evidence you need to decide whether to use it.

<details>
<summary>Check your answers</summary>

1. Each independent request supplies its own ready input. Sequential dependence remains within each request.
2. 8/0.020 = 400 tokens/s; each request's ITL is 20 ms. 2.5 ms is amortized step cost per output token, not observed ITL.
3. A finished, freeing a slot that the scheduler may refill at the boundary. C was already prefilled and ready to decode.
4. The timeout limits deliberate accumulation in this policy, not waiting for a busy worker or an overloaded queue.
5. No. The next request is still 85 ms away when the deadline expires. The batch stays size one and latency rises from 8 to 23 ms in this example.
6. 80 × 0.250 = 20 requests on average, including waiting requests. Batch size is not determined by that average.
7. 16 GiB, under the same fixed-layout assumptions. Weights, workspaces, other activations, allocation overhead, and further cache growth are excluded.
8. More rows can improve weight reuse while lengthening steps, and assembly or scheduling can delay first tokens. Compare the same workload, actual batch sizes, offered/achieved load, memory, rejection rates, latency distributions, and fraction meeting the required SLO. Choose using the application's requirements.

</details>

## Interview notes

- Batching exposes independent rows that can reuse loaded weights; arithmetic and request-specific state remain.
- Concurrency counts outstanding requests; batch size counts grouped execution work.
- Dynamic formation collects work before a call. Continuous scheduling changes membership between generation iterations.
- Throughput is aggregate work per time; amortized cost is not individual latency.
- Evaluate queueing, tail latency, cache capacity, and admitted workload alongside execution speed.

**Next lesson:** Quantization and memory use.
