# Lesson 2: Prefill, decode, and the KV cache

[Previous: Metrics](01-inference-metrics.md) · [Course overview](../README.md) · [Next: Bottlenecks](03-compute-memory-bottlenecks.md)

**Status:** Complete. This walkthrough assumes an ordinary causal decoder-only transformer with full attention, one request, no speculative decoding, and no reused prompt cache. Tokenization and output choices below are illustrative.

## Follow one request from the beginning

Prompt: **what is the capital of france**

Assume six tokens, with input representations x1 through x6. Real tokenizers may split the text differently. A representation is a vector of numbers, not the written word itself.

```text
Position:     1      2      3       4       5      6
Token:       what    is    the   capital   of   france
Input:        x1    x2     x3      x4      x5     x6
```

## 1. Prefill processes the prompt

The six inputs pass through the model's layers. In an attention layer, the model calculates queries, keys, and values. Using row-vector notation:

```text
q_i = x_i W_Q
k_i = x_i W_K
v_i = x_i W_V
```

These equations refer to the inputs at that particular layer. Later layers receive representations transformed by earlier layers. Each layer generally has its own projection matrices. Position-related operations and other attention details are omitted here to keep the lifecycle clear.

The layer keeps the keys and values for each prompt position. Those retained entries form its KV cache.

Prompt positions can undergo substantial parallel processing within a layer because their inputs are already available. Causal attention still prevents a position from attending to future positions. Layers run in dependency order, and finite hardware means a longer prompt does not take constant time.

## 2. The end of prefill predicts the first output token

The final model representation at the last prompt position is used to produce **logits**: scores for possible next tokens. A selection rule, such as sampling or choosing the highest-scoring token, selects the first output token.

Suppose it selects **Paris**.

At this moment:

- Paris can be returned to the user.
- Each layer has keys and values for the six prompt positions.
- Paris has been selected, but has not yet been processed as a new input position.

**Prefill produces both the prompt cache and the scores used to select the first output token.** A separate forward pass over Paris is not needed to select Paris itself.

## 3. Decode processes Paris to predict what follows

Feed Paris back into the model at position 7.

At each attention layer:

1. Calculate this position's query, key, and value.
2. Add its key and value to that layer's cache.
3. Use its query to attend to keys and values for the previous positions and its own position.
4. Continue through the layer and the rest of the model.

The final output produces scores for the following token. Suppose the selected token is a period.

After this forward pass, all layers have seven positions cached. The period has been selected but not yet processed as an input.

## 4. Repeat until a stopping condition

Process the period at position 8. This adds its per-layer keys and values and produces scores for the next token. Suppose the selected token is EOS, an end-of-sequence marker. Stop generation; there is no need to feed EOS back through the model.

```text
Process six prompt tokens → select Paris
Process Paris             → select .
Process .                 → select EOS → stop
```

| Moment | Newly selected token | Cached positions per layer |
| --- | --- | ---: |
| End of prefill, first token selected | Paris | 6 |
| After processing Paris | . | 7 |
| After processing the period | EOS | 8 |

Output-token accounting can include or exclude EOS depending on the system; state the convention when measuring.

## If there are 34 layers, are there 34 caches?

If all 34 layers use cached attention, each has its own cached keys and values. We can think of 34 logical per-layer KV caches, collectively called the KV cache.

Each layer stores its own representations of the **same sequence positions**. Six positions at 34 layers is still a six-token sequence, not a 204-token sequence. Implementations may pack these logical caches into larger memory structures.

Processing Paris extends each layer's cache from six to seven positions as the token passes through that layer.

## What the cache saves—and what it doesn't

The model reuses previous keys and values rather than recalculating them from all previous tokens at every step. New-token attention still needs to access relevant cached entries. With full attention, a longer context means more prior positions to attend to; the cache is not a promise of constant-time decode.

This is distinct from caching a completed answer or reusing a prompt prefix across separate requests.

## Compare two workloads

| Request | Prompt | Output | Interpretation |
| --- | ---: | ---: | --- |
| A | 4,000 tokens | 20 tokens | More prefill work; long context at generation start |
| B | 20 tokens | 4,000 tokens | More sequential generation steps; context grows substantially |

Anand correctly identified these differences. Same model and backend do not guarantee equal ITL: context length and serving conditions also matter. Token counts alone do not establish a precise total-runtime ordering.

## Recall before moving on

1. Immediately after Paris is selected from prefill's logits, does each layer cache six or seven positions?
2. Why is a new forward pass needed after selecting Paris?
3. Does the KV cache eliminate reading previous keys and values?

<details>
<summary>Check your answers</summary>

1. Six. Anand answered this checkpoint correctly.
2. Processing Paris produces the scores for the token after Paris and adds Paris's own cache entries.
3. No. It saves recalculation; attention still accesses relevant cached data.

</details>

**Takeaway:** Processing current inputs predicts the next token. Selecting a token and processing that token are different events.
