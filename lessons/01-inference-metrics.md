# Lesson 1: Inference metrics

[Course overview](../README.md) · [Next: Prefill and decode](02-prefill-decode-kv-cache.md)

**Status:** Complete. All examples here are hypothetical timing examples, not model benchmark results.

## What does faster inference mean?

A response can start sooner, produce later tokens more quickly, or allow a server to handle more total work. Those are different improvements. Specify the metric before claiming a speedup.

| Metric | Meaning | Unit |
| --- | --- | --- |
| Time to first token (TTFT) | Elapsed time from a defined request start to the first output token | seconds or milliseconds |
| Inter-token latency (ITL) | Time between successive output tokens | seconds or milliseconds per interval |
| Total response time | Time from request start to the final output token in these examples | seconds or milliseconds |
| Throughput | Work completed per time interval; specify the workload and measurement window | e.g. tokens/second |

In a real system, request start might mean client submission or server receipt. End-of-request events can occur after the last visible token. Keep those boundaries explicit and consistent.

## Worked example: count the gaps

A request begins at time zero. Its first token arrives after 0.6 seconds. It produces six output tokens, with 0.08 seconds between successive arrivals.

```text
Token:       1       2       3       4       5       6
Arrival:   0.60    0.68    0.76    0.84    0.92    1.00 seconds
           first                                    final
```

There are six tokens but only five gaps after the first one.

```text
TTFT = 0.6 seconds
ITL  = 0.08 seconds
Final-token time = 0.6 + (6 - 1) × 0.08 = 1.0 seconds
```

For N output tokens and constant interval d:

```text
Total response time = TTFT + (N - 1) × d
```

For variable intervals, add the actual N - 1 intervals instead. Real token spacing need not be constant.

## Two different token rates

The subsequent-token rate describes the spacing after the first token:

```text
Subsequent-token rate = 1 / 0.08 = 12.5 tokens/second
```

Equivalently, five additional tokens arrive over the 0.4 seconds between first and last arrival: 5 / 0.4 = 12.5.

The whole-request output rate includes the initial wait:

```text
Whole-request output rate = 6 / 1.0 = 6 tokens/second
```

Both calculations are valid. They use different windows. Neither automatically describes aggregate server throughput across concurrent requests.

With only one output token, there is no inter-token interval, so this observed subsequent-token rate is undefined.

## Improving TTFT alone

Suppose TTFT drops to 0.3 seconds, while output length and ITL stay unchanged:

```text
New total response time = 0.3 + 5 × 0.08 = 0.7 seconds
```

The response finishes earlier. Its subsequent tokens remain 0.08 seconds apart, and their rate remains 12.5 tokens/second. The whole-request output rate improves because its denominator becomes smaller.

## Reviewed work

Anand correctly calculated 12.5 tokens/second and explained that improving TTFT alone need not change ITL. The initial final-token answer of 0.6 seconds came from reading the question as asking for the first token; the corrected final-token time is 1.0 second. This was clarified as a reading slip.

## Recall before moving on

1. Why do six tokens have five intervals after the first token?
2. With first arrival at 0.4 seconds and five tokens spaced 0.1 seconds apart, when does the last arrive?
3. Why can a request have a subsequent-token rate of 10 tokens/second but a lower whole-request rate?

<details>
<summary>Check your answers</summary>

1. The first token ends the initial waiting period. Each of the remaining five arrivals adds one interval.
2. 0.4 + (5 - 1) × 0.1 = 0.8 seconds.
3. The whole-request rate also includes the wait before the first token. For this example it is 5 / 0.8 = 6.25 tokens/second.

</details>

**Takeaway:** Define the timing window, count intervals correctly, and name the metric you want to improve.
