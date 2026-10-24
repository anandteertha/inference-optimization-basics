# Learning progress

Last reviewed: October 24, 2026 (America/New_York).

## Current position

**Lessons 1-3 complete. Next: Lesson 4, batching and serving tradeoffs.** Lesson 4 has not started. Coding begins after all eight conceptual lessons; Project 1 / Lab 1 is the token timing analyzer.

## Reviewed understanding

| Lesson | Evidence | Status |
| --- | --- | --- |
| 1. Inference metrics | Correctly calculated 12.5 tokens/second from 0.08-second ITL; distinguished TTFT improvements from unchanged ITL. First/final-token wording slip clarified. | Complete |
| 2. Prefill, decode, KV cache | Correctly identified longer-prompt prefill work and longer-output sequential generation; explained cached K/V; correctly answered six cached positions after selecting Paris and before processing it. Per-layer cache structure clarified. | Complete |
| 3. Compute and memory bottlenecks | Explained capacity versus bandwidth and compute; connected fixed projection matrices to weight reuse; reasoned about arithmetic per fetch; correctly solved compute- and memory-limited overlap examples. | Complete |
| 4-8. Remaining concepts | Not yet taught or reviewed. | Upcoming |
| Projects 1-4 / Labs 1-4 | No coding projects started. | Upcoming |

## Final Lesson 3 answers

For 2 ms arithmetic and 8 ms memory transfer, with ideal complete overlap:

1. Baseline: max(8, 2) = 8 ms, memory-bandwidth-bound.
2. Double compute throughput: 1 ms arithmetic, max(1, 8) = 8 ms, still memory-bandwidth-bound.
3. Double memory bandwidth from baseline instead: 4 ms transfer, max(4, 2) = 4 ms, still memory-bandwidth-bound.

All three answers were independently provided with correct reasoning.

## Clarifications worth revisiting

- The prompt's final position produces scores used to select the first output token. Selecting a token and processing it as input are different events.
- A 34-layer model with cached attention has distinct per-layer K/V entries for the same sequence positions.
- Shared projection weights mean the same matrix is applied across token positions in a layer. They do not imply identical matrices across all layers.
- Mathematical matrix multiplication does not guarantee the full weight matrix is fetched once. Hardware can reuse loaded pieces across rows.
- Decode has less opportunity for reuse across new input rows within one request's step; this does not imply universally zero weight reuse.
- Overlap is an explicit idealization, not proof that every pair of GPU tasks runs simultaneously.

## Teaching agreement

- Understanding takes priority over finishing the theory or meeting a schedule.
- Preserve the eight-lesson conceptual sequence before coding.
- Define a term before drawing conclusions from it.
- Keep the same concrete example while explaining a causal chain.
- Use the learner's existing matrix knowledge instead of adding unnecessary analogies.
- Ask one focused checkpoint at a time; review the answer before introducing the next connection.
- A typical session can last 45-60 minutes, but a lesson may take multiple sessions.

## Next session starting point

Begin Lesson 4 with two requests, each processing one new token in decode using the same model. Explain what it means to group their input rows and reuse weights across them. Check that mechanism before discussing throughput, request latency, queueing, or continuous batching.

For a refresher first, read [the quick reference](REFERENCE.md). The [README](README.md) contains the full course roadmap independently of this progress record.
