---
title: VLLM
one_liner: Trace and edit a model whose forward pass is a vLLM engine — continuous batching, flat token rows, per-step values, CUDA-graph taps.
tags: [models, vllm, serving, production]
related: [docs/models/index.md, docs/models/transformers-model.md, docs/models/vllm-editing.md, docs/models/vllm-serving.md, docs/models/vllm-parallelism.md]
sources: [src/nnsight/modeling/vllm/vllm.py, src/nnsight/modeling/vllm/tracer.py, src/nnsight/modeling/vllm/interleaver.py, src/nnsight/modeling/vllm/batching.py, src/nnsight/modeling/vllm/fragments.py, src/nnsight/modeling/vllm/collect.py, tests/vllm/]
---

# VLLM

## What this is for

`nnsight.modeling.vllm.VLLM` runs NNsight interventions on top of vLLM's inference engine. The model runs in vLLM's own worker process(es); your intervention code is serialized onto each request (via `SamplingParams.extra_args`), carried into the worker, run against the real module, and the saved values shipped back. You get PagedAttention, continuous batching, tensor parallelism, and optional async streaming, with arbitrary Python interventions inline with the forward.

Interventions are written exactly as for any other model — the module tree mirrors what vLLM loaded. The one structural difference: each `tracer.invoke(...)` is **one vLLM request** (one prompt), so several prompts means several invokes, not a list.

This page is the engine and what a block sees inside it. Three companions carry the rest:
[Editing a vLLM engine](vllm-editing.md) for `model.edit()`, [Async mode and
serving](vllm-serving.md) for `mode="async"` and `nnsight-serve`, and [Parallelism and
architectures](vllm-parallelism.md) for tensor parallelism, mixture-of-experts and hybrid trunks.

Running this needs `vllm` installed and a GPU.

Use `VLLM` for throughput over many concurrent prompts, tensor parallelism, or streamed tokens
with interventions. For a single prompt, gradients, `.scan()`, or image inputs, use
[`TransformersModel`](transformers-model.md).

## Loading

### Constructor

```python
VLLM(
    repo_id,
    *,
    mode="sync",                  # "sync" (vllm.LLM) or "async" (AsyncLLM)
    dispatch=False,               # True = build the engine now
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    taps=(),                      # locations to serve under CUDA graphs; empty = eager, every location
    **vllm_kwargs,                # forwarded to vllm.LLM / AsyncEngineArgs
)
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | HuggingFace repo id. |
| `mode` | `"sync"` (default) builds a `vllm.LLM` and runs synchronous generation; `"async"` builds a `vllm.v1.engine.async_llm.AsyncLLM` and a trace streams outputs via `async for output in tracer.backend`. Set at construction — you can't switch per trace. |
| `dispatch` | `True` creates the real engine during `__init__`. `False` (default) builds only a meta-tensor tree (via vLLM's `DummyModelLoader`, `device="meta"`) — no GPU memory used until the first trace. |
| `tensor_parallel_size` | GPUs to shard across. Transparent to your interventions (see [Parallelism and architectures](vllm-parallelism.md)). |
| `gpu_memory_utilization` | vLLM KV-cache budget (default 0.9). Lower it (e.g. 0.1) for small models / shared GPUs. |
| `taps` | Module locations to serve with CUDA graphs **on** — `"model.layers.*.output"`, `*` one path segment, or a layer's [kernel sites](#attention-kernel-sites), `"model.layers.*.self_attn.attn.kernel"`. Empty (default) runs the engine eagerly, where every location is served. See [CUDA graphs with taps](#cuda-graphs-with-taps). |
| `**vllm_kwargs` | Anything valid for `vllm.LLM` / `AsyncEngineArgs`. |

The NNsight worker class is always forced internally, and so is `enforce_eager=True` unless you declare `taps` — CUDA graphs freeze the ops they replay, so an ordinary hook can't fire inside one.

## Canonical pattern (sync)

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True)

with model.trace("The Eiffel Tower is located in the city of", temperature=0.0, top_p=1):
    model.transformer.h[8].output[:] = 0        # intervene
    logits = model.logits.save()

print(model.tokenizer.decode(logits.argmax(dim=-1)))
```

### What your block sees on vLLM

What differs from the HuggingFace path:

- **No batch axis.** A served value is `[tokens, hidden]` for *your* request's rows: the prefill step
  serves every prompt token, each later decode step serves one row. The last position is `[-1]`,
  never `[:, -1, :]`.
- **A decoder layer's `.output` is a pair `(hidden, residual)`** on Llama/Qwen/Mistral-style models,
  because vLLM fuses the residual add into the next layer's norm. `output[0]` is this layer's
  sub-block output (norm ~30 on Qwen3-8B), `output[1]` the residual stream entering it (norm ~1000);
  the residual stream *after* the layer is their sum. Patching `output[0]` alone changes little;
  writing either element steers, since the next norm adds them.

  ```python
  out = model.model.layers[20].output
  resid = (out[0] + out[1]).clone()        # residual stream after layer 20, [tokens, hidden]
  out[0][-1] += v                          # additive steering at the last position
  ```

  Scale a steering vector against the *median* per-token residual norm: the first
  position is often an attention sink with a norm ~100× the rest, and on 8B-class
  models a coefficient of ~1× that median already saturates.

- **Clone what you keep.** A served value is the engine's live buffer and is overwritten a few ops
  later; `.clone()` or reduce before you save. See [Serving copies instead of
  views](#serving-copies-instead-of-views).
- **Where tensors live.** A tensor your block references from outside (a steering vector) is
  serialized with the block and arrives in the worker as it was; move it onto the served value
  (`v.to(h.device, h.dtype)`). Saved tensors come back on the worker's device (`cuda:0`) unless
  you `.cpu()` them in the block. `model.device` on an undispatched client is `meta`.
- **A request's numbers depend on the batch it was scheduled with.** Fused-kernel reduction order
  follows the batch, so in bf16 the same greedy prompt can flip a nearly tied argmax and a sweep
  reproduces to about a part in a hundred. `temperature=0.0` pins the sampler, not the arithmetic:
  compare effect sizes, and read a paired difference (clean against patched *in the same trace*)
  rather than differencing two runs.

`model.logits` is `[1, vocab]` for the step and `model.samples` is `[1, 1]` (`.item()` for one
sequence). Greedy decoding is `temperature=0.0`; vLLM's default is `1.0`.

### Replacing a value keeps its rows

An in-place write (`out[0][-1] += v`, `mlp.output[:] = 0`) always fits. A *replacement* —
`layer.output = t` — is spliced back into the running batch as given, so it must carry the same
rows the block owns; nothing checks. A wrong leading dimension can surface as a device-side assert,
which poisons the CUDA context and takes the engine and every request in it. In a patching sweep
with a donor from a different prompt length, slice to the rows you write
(`served[POS] = donor[POS]`) rather than handing back a shorter tensor.

### `.skip()` needs the whole step

A skip has to cover **every row of the step**, and a step's rows include other traces', other
clients', and finished requests' decodes. The moment anything is scheduled beside you,
`ValueError: A batched .skip() has to cover every row` is raised outside the per-request deferral,
so vLLM treats it as fatal: `EngineDeadError`, every in-flight request lost, the model object
unusable. Whether a script survives is the scheduler's decision, so do not build on it — write the
output instead (`model.model.layers[L].mlp.output[:] = 0`).

A decoder layer's `.input` is not the hidden state either: vLLM calls
`forward(positions, hidden_states, residual)`, so `layer.input` is the positions vector and
`module.skip(module.input)` is fatal too. Read the arguments through `.inputs`
(`args, kwargs = layer.inputs`; `args[1]` is the hidden state, `args[2]` the residual).

### `logits` and `samples`

Served values on the model (`eproperty` descriptors), meaningful only inside a trace. For the finished request as a whole, read `tracer.result`.

| Property | Description |
|----------|-------------|
| `model.logits` | The pre-sampling logit tensor for this step. Read or assign (`model.logits = ...`). On a request with `structured_outputs`, vLLM has already applied its grammar mask: disallowed tokens read `-inf`. |
| `model.samples` | The token ids the sampler drew from `logits` this step. Assigning it forces the token the engine continues from. |

Under `tracer.iter`, each pass sees the next decoded step.

### Multi-token generation

```python
with model.trace("Madison Square Garden is located in the city of",
                 temperature=0.0, top_p=1.0, max_tokens=3) as tracer:
    logits = list().save()
    for _ in tracer.iter[0:3]:
        logits.append(model.logits)

print(model.tokenizer.batch_decode([l.argmax(dim=-1) for l in logits]))
# [' New', ' York', ' City']
```

Step 0 is the prefill; `model.samples` on step *k* is the *k*-th generated token
(`result.outputs[0].token_ids[k]`). Plain Python locals persist across steps, so a block can keep
state — a flag flipped by a probe at one step steers every later one. A write to a layer *below*
a read lands on the next step, not the current one.

`tracer.all()` iterates every generated step:

```python
with model.trace(PROMPT, max_tokens=10) as tracer:
    logits = list().save()
    for _ in tracer.all():
        logits.append(model.logits)
# len(logits) == 10
```

#### A loop that outruns the request

`max_tokens` is a cap, not a promise: EOS, a `stop` string, `stop_token_ids` or a `tracer.stop()`
elsewhere end a request early, and a `tracer.iter` loop that asked for more steps is cut short
there, as described in [iteration.md](../usage/iteration.md). On vLLM the warning is emitted in the
EngineCore subprocess (see [Where an error comes from](#where-an-error-comes-from)), so nothing in
your process says so:

```python
with model.trace(PROMPT, temperature=0.0, max_tokens=20, stop=[","]) as tracer:
    samples = list().save()
    for _ in tracer.iter[:20]:
        samples.append(model.samples.item())
    result = tracer.result.save()          # never reached: `result` comes back unbound
# len(samples) == 4
```

Check the `len()` of what you collected, hold the run to the count with `ignore_eos=True` or
`min_tokens=N` (vLLM's spelling; the warning's `min_new_tokens=` is refused here), or loop with
`tracer.all()` and put trailing statements in a separate `tracer.invoke()`.

A request that ends on a stop token or a `stop` string can also give a block one step more than
its output holds. vLLM schedules a request's next step before the last step's token is known
(`async_scheduling`, on by default), so the forward after the stop still runs, and a block under
`tracer.iter` or `tracer.all()` sees it. A request that ends at `max_tokens` is exact. Where the
two counts must agree, count against `len(result.outputs[0].token_ids)`, or build the engine with
`VLLM(..., async_scheduling=False)`, which makes a stop token exact at a cost in throughput; a
`stop` string can still overrun, since it is matched on the decoded text after the step.

### `generate` traces, or just runs

vLLM generation is driven by `max_tokens`, so there is no forward/generate split. Used as a `with` block, `model.generate(...)` is `trace` (and rewrites `max_new_tokens` → `max_tokens`; `trace` accepts either).

Called **without** a `with` block it simply runs the engine and returns vLLM's `RequestOutput`s — which is how you read an edit's values without reaching past the model for `model.vllm_entrypoint`:

```python
outputs = model.generate(prompts, max_tokens=5)
outputs[3].saves["hidden"]        # see Editing a vLLM engine
```

On `mode="async"` the same call returns an awaitable: `outputs = await model.generate(...)`.

### `tracer.result`

`tracer.result` is the finished `RequestOutput` for the request an invoke made — one per invoke:

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=3) as tracer:
    out = model.model.layers[8].output
    hidden = (out[0] + out[1]).clone().save()
    result = tracer.result.save()          # last: nothing can be read after it

result.outputs[0].text        # ' Paris, France'
```

It is served at collect time, after every other visit of the request, so it must be the **last**
read in the block (a later read raises `OutOfOrderError`). It carries the generation but **not**
`.saves`; per-step values come from `model.logits` / `model.samples` under `tracer.iter`.

### Sampling parameters

Sampling kwargs go to `trace`/`invoke` (not configured on the model) and become the request's `SamplingParams`:

```python
# root-level params apply to all invokes
with model.trace("Hello", temperature=0.7, top_p=0.95, max_tokens=10) as tracer:
    samples = list().save()
    for _ in tracer.iter[:]:
        samples.append(model.samples.item())

# per-invoke params override root
with model.trace(max_tokens=3) as tracer:
    with tracer.invoke("Hello", temperature=0.0, top_p=1.0):
        ...
    with tracer.invoke("Hello", temperature=1.5, top_p=0.95):
        ...
```

Anything `vllm.SamplingParams` takes is accepted (`temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`, `logprobs`, `lora_request`, ...). Per-invoke kwargs override trace-level ones even when the value an invoke names is vLLM's own default (`temperature=1.0`); a typo'd kwarg raises `TypeError`.

### Input forms per invoke

One prompt per invoke: a string, a list of token ids, a tokenizer's `{input_ids, attention_mask}` output, or one of vLLM's own prompt dicts (`TokensPrompt`, `TextPrompt`). A list of strings / multiple prompts is rejected — use one invoke each.

### Continuous batching (multiple invokes)

Each invoke is one request; vLLM's continuous batcher processes them together.
Your block runs once per request, so a name saved in every invoke comes back as a
**list**, one entry per invoke in order:

```python
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for prompt in prompts:
        with tracer.invoke(prompt):
            hidden = model.transformer.h[5].output.save()

hidden[0]        # the first invoke's, sized to its own prompt
len(hidden)      # == len(prompts)
```

A name saved by exactly one invoke is that value, not a list of one, so a distinct name per invoke also works.

A container bound *and saved* **above** the invokes is one object locally, and each request gets
its own copy: assigning into pre-sized slots merges back, `append` does not.

```python
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    rows = nnsight.save([None, None])
    with tracer.invoke(prompt_a):
        rows[0] = model.transformer.h[5].output
    with tracer.invoke(prompt_b):
        rows[1] = model.transformer.h[5].output
```

### Passing values between invokes

Each invoke's block runs on its own request, in its own scope: a value one invoke
reads is **not** visible to a sibling (`NameError`), `tracer.barrier` is
unavailable, and a `session()` does not bridge traces either. Hand values across
with two traces — a saved value from the first ships with the block of the second:

```python
with model.trace(clean, temperature=0.0, max_tokens=1):
    donor = nnsight.save(tuple(o.clone() for o in model.model.layers[L].output))

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    with tracer.invoke(corrupt):
        for served, saved in zip(model.model.layers[L].output, donor):
            served[POS] = saved[POS].to(served.device)
        logits = model.logits.save()
```

Many invokes in one trace still batch: a layer sweep is one trace with one
patched invoke per layer.

### Several sampled sequences (`n > 1`)

`n=k` asks vLLM for k continuations of one prompt, and it fans the request into a
child per sequence — so your block runs k times, once per sequence, against that
sequence's own rows. Every saved name comes back as a list here for the same reason,
`tracer.result` included:

```python
with model.trace(prompt, max_tokens=5, temperature=1.0, n=3) as tracer:
    hidden = model.transformer.h[5].output.save()
    result = tracer.result.save()

hidden[1]                        # sequence 1's
result[1].outputs[0].text        # what sequence 1 sampled
```

Each `result[i]` is a `RequestOutput` holding sequence *i*'s completion, with whatever an
installed edit saved for that sequence on `result[i].saves`. With several invokes too, the list
runs invoke by invoke, then sequence by sequence: `[a0, a1, b0, b1]` for two invokes at `n=2`.

Where you hold outputs rather than variables (async, plain `generate` with an edit), sequence
*i*'s values are `output.outputs[i].saves[...]`; `output.saves` is the primary sequence's. On an
**async** engine each streamed output carries only the completions that advanced that step —
accumulate by `completion.index`, or read the finished output's `nnsight_sequences`.

### LoRA adapters

Build the engine with `enable_lora=True` and name an adapter per request with vLLM's
`LoRARequest`. The adapter is a PEFT directory (`adapter_config.json` and
`adapter_model.safetensors`).

```python
from vllm.lora.request import LoRARequest

model = VLLM("meta-llama/Llama-3.2-1B", dispatch=True, enable_lora=True, max_lora_rank=8)
adapter = LoRARequest(lora_name="mine", lora_int_id=1, lora_path="/path/to/adapter")
model.vllm_entrypoint.llm_engine.add_lora(adapter)   # load it now; raises if it cannot

with model.trace(temperature=0.0, max_tokens=1, lora_request=adapter) as tracer:
    with tracer.invoke(prompt):                       # the trace's adapter
        adapted = model.logits.save()
    with tracer.invoke(prompt, lora_request=None):    # opts out: the base model
        base = model.logits.save()
```

- `lora_request=` goes on `trace`, `invoke` or a plain `generate`. On `trace` it is the default
  for its invokes; an invoke that names its own, `None` included, keeps it.
- `max_lora_rank` (vLLM's default is 16) has to cover the adapter's rank. `max_loras` (default 1)
  is how many different adapters one step can hold.
- A LoRA-wrapped layer's `.output` is the adapted value.
- **Load the adapter with `add_lora` first.** A request naming an adapter the engine cannot load
  (a missing path, a rank over `max_lora_rank`) kills the engine with `EngineDeadError`, as it
  does in plain vLLM. `add_lora` raises an ordinary exception carrying the real message, and the
  engine keeps serving.
- Without `enable_lora`, a request naming an adapter raises `VLLMValidationError: Got lora_request
  ... but LoRA is not enabled!`.
- Over `serve=`, start the server with `--enable-lora True`; `lora_path` is read on the server.

## Special properties

| Attribute | Description |
|-----------|-------------|
| `model.logits` / `model.samples` | Pre-sampling logits / sampled ids for the step (trace-only, `eproperty`). |
| `model.tokenizer` | The tokenizer vLLM resolved for the checkpoint. Loaded eagerly. |
| `model.vllm_entrypoint` | The underlying `vllm.LLM` (sync) or `AsyncLLM` (async); `None` until dispatch. |
| `model.taps` | The resolved tap set on a `taps=` engine; empty on an eager one. |
| `model.dispatched` | Whether the engine is built (`MetaMixin`). |

### Module structure

The Envoy tree mirrors vLLM's model layout. For Llama-style models:

```python
model.model.layers[i].self_attn.qkv_proj.output   # ColumnParallelLinear
model.model.layers[i].self_attn.o_proj.output     # RowParallelLinear
model.model.layers[i].mlp.gate_up_proj.output
model.model.layers[i].mlp.down_proj.output
model.model.norm.output
```

`model.model.layers[i].output` is the `(hidden, residual)` pair described under
[What your block sees](#what-your-block-sees-on-vllm). For GPT-2-style models: `model.transformer.h[i].attn.output`, `model.transformer.h[i].mlp.output`. MoE blocks carry `mlp.gate`, `mlp.experts` (and on Qwen-MoE `mlp.shared_expert`); hybrid trunks carry `linear_attn` or `self_attn` per layer; vision-language checkpoints keep the decoder at `model.language_model.model.layers`. Print `model` (or one layer, `model.model.layers[0]`, on a large checkpoint) to see the actual tree.

Paths from `model.named_modules()` carry the root's name (`model.model.layers.0`); `model.get(...)` and `taps=` take the path without it (`model.layers.0`), and `model.taps` reports the prefixed form.

## Attention kernel sites

On vLLM's FlashAttention backend (the default on CUDA), the inner attention
layer carries the [kernel sites](../usage/kernel.md) `lse`, `scores` and
`probs`. The layer is the `Attention` module inside the model's attention
block: `model.transformer.h[i].attn.attn` on GPT-2, `model.model.layers[i].self_attn.attn`
on Llama-style models.

```python
attn = model.transformer.h[0].attn.attn
with model.trace("The Eiffel Tower is in", max_tokens=3, temperature=0.0) as tracer:
    steps = list().save()
    for _ in tracer.iter[:3]:
        steps.append(attn.kernel.probs().clone())
# [tuple(p.shape) for p in steps] == [(7, 12, 7), (1, 12, 8), (1, 12, 9)]  (the prompt is 7 tokens)
```

Shapes follow vLLM's layout: there is no batch axis, and an invoke sees only
its own request.

| Site | Shape | Value |
|------|-------|-------|
| `lse` | `[queries, heads]` | The logsumexp the kernel computed for each query row, fp32. |
| `scores` | `[queries, heads, keys]` | `q . k * scale` in fp32, `-inf` above the causal diagonal. |
| `probs` | `[queries, heads, keys]` | `exp(scores - lse)`, fp32. |

The query axis holds the rows the engine computed in this step: every prompt
token on the prefill step, one token on each decode step. The key axis always
spans every token of the request so far, read from the request's pages of the
KV cache. `query=`, `heads=` and `key=` select along each axis with the rules of
[kernel sites](../usage/kernel.md#sites), and a selection is applied before the
matrix is built, so `probs(query=-1, heads=3)` costs one row.

The values are rebuilt from the kernel's query rows, the cached keys and the
kernel's own logsumexp, which the kernel returns from the same launch that
computes the attention output. On GPT-2 layer 0 they equal the HF site's
`probs` in bf16 to 1e-5. Deeper layers differ from HF by the bf16 rounding the
two engines accumulate in the residual stream.

A call the reconstruction does not model raises `ValueError` with the reason
at the request: a sliding window, ALiBi, a logit soft cap, attention sinks, a
custom mask, an fp8 KV cache, or a step vLLM computes with cascade or
context-parallel attention (several kernel calls merged into one).
Under tensor parallelism each rank's kernel call holds a block of the heads.
A site still returns every head, and `heads=` indexes the model's heads: each
rank fills the heads it holds and one sum over the ranks assembles the value
([the reader contract](../usage/kernel.md#the-reader-contract-axes-and-their-layout-across-ranks)).
On Qwen2.5-0.5B at `tensor_parallel_size=2`, layer 0 equals the one-rank values
to 1e-5.

On an engine built with `taps`, declare the kernel sites as a tap, by the
attention layer's path with `.kernel` on the end:

```python
model = VLLM("gpt2", dispatch=True, taps=["transformer.h.*.attn.attn.kernel", "transformer.h.*.output"])
```

The sites are then served on every replayed step with the values of the eager
engine (equal on GPT-2 over a prefill and two decode steps). vLLM runs its
attention op in Python on every replay, between graph segments, so a kernel tap
adds no graph break of its own to the attention call; it records two small
callables around the layer. A kernel site that is not a tap raises the same
"is not a tap on this engine" error as any other untapped location.

The sites are not served on other attention backends (Triton, FlashInfer, MLA).

## MLA attention kernel sites

On a model with multi-head latent attention (DeepSeek-V2 and related), the
inner attention module carries the same three sites, with the same
`[query, heads, key]` layout and no batch axis:

```python
attn = model.model.layers[3].self_attn.mla_attn.mla_attn     # DeepSeek-V2-Lite
with model.trace(prompt, max_tokens=3, temperature=0.0) as tracer:
    steps = list().save()
    for _ in tracer.iter[:3]:
        steps.append(attn.kernel.probs(query=-1).clone())    # [heads, keys so far]
```

An MLA layer runs a step's tokens through two kernels, and the sites follow the
one a request went through:

| Step | Kernel | `scores` | `lse` | `probs` |
|------|--------|----------|-------|---------|
| Prefill | FlashAttention over the new tokens' expanded keys | `q . k * scale` in fp32 | the kernel's, fp32, requested in the same launch | `exp(scores - lse)` (exact reconstruction) |
| Decode | Triton decode kernel over the request's compressed rows in the paged cache | `q . k * scale` in fp32, with the query projected into the compressed space as the kernel receives it | the kernel's, in the query dtype | fp32 softmax of the scores (`replay`), because a bf16 logsumexp is too coarse to divide by |

On DeepSeek-V2-Lite the probabilities of layers 0 and 3 equal transformers'
eager attention to within 0.07 at prefill and 0.03 at two decode steps, with
both in bf16; against an fp32 transformers run the gap is 0.035 and 0.011.
The sampled tokens are identical. Under `tensor_parallel_size=2` the sites
assemble the ranks' heads and equal the one-rank engine to within 0.05 (the
test's bound).

On an engine built with `taps`, name the site as
`"model.layers.*.self_attn.mla_attn.mla_attn.kernel"`. Layer 0 then equals the
eager engine to 1e-5 and layer 7 to 2.5% of the largest entry, the drift between
a replayed graph and eager execution.

Not served: a prefill that continues cached or earlier-chunk context (prefix
caching or chunked prefill), which vLLM computes as several kernel calls merged
by logsumexp; the other MLA backends (FlashMLA, FlashInfer, CUTLASS, sparse
variants); and a quantized cache.

## Recurrent kernel sites

On Qwen3-Next and Qwen3.5 models, a gated delta rule layer
(`layers[i].linear_attn`) carries the recurrent
[kernel sites](../usage/kernel.md#recurrent-states-of-a-chunked-kernel)
`chunk_states` and `states`:

```python
attn = model.language_model.model.layers[0].linear_attn      # Qwen3.5-0.8B
with model.trace(prompt, max_tokens=3, temperature=0.0) as tracer:
    steps = list().save()
    for _ in tracer.iter[:3]:
        steps.append(attn.kernel.chunk_states().clone())
# a 145-token prompt: [tuple(s.shape) for s in steps] == [(4, 16, 128, 128), (2, 16, 128, 128), (2, 16, 128, 128)]
```

| Site | Shape | Value |
|------|-------|-------|
| `chunk_states` | `[chunks, heads, K, V]` | On a prefill step: the state entering each 64-token chunk of the request's tokens in this step, then its final state. On a decode step: the state before the token and the state after it. |
| `states` | `[tokens, heads, K, V]` | The state after each token of the step. A token that ends a chunk is the kernel's own state; one inside a chunk is advanced from the chunk's entering state in fp32. |

There is no batch axis, and an invoke reads only its own request. `chunks=`,
`tokens=` and `heads=` select as on a transformers model, and a selection is
applied before anything is computed.

vLLM's kernels and its state cache hold a state as `[heads, V, K]`. The sites
serve `[heads, K, V]`, the layout of the transformers sites, so the same code
reads both. On Qwen3.5-0.8B layer 0, the chunk states and the per-token states
of a 145-token prompt equal the transformers sites to 0.8% of the largest entry;
vLLM's kernel runs in bf16 and transformers' reference in fp32.

Each request's state lives in a slot of the layer's state cache, and a decode
step advances the slot in place. The state a decode step serves as entering is
therefore the state the previous step served as final, bit for bit.

Under tensor parallelism each rank's layer holds `num_v_heads // tp_size`
value heads and a state of that many heads. The sites assemble the ranks'
blocks, so `heads=` indexes the model's heads. On Qwen3.5-0.8B at
`tensor_parallel_size=2`, layer 0 equals the one-rank engine exactly on a
prefill step and to 3e-5 on a later decode step.

On an engine built with `taps`, name the site as
`"language_model.model.layers.*.linear_attn.kernel"`. Layer 0 then equals the
eager engine exactly. Deeper layers differ by a few percent of the largest
entry because the layer outputs feeding them already differ between a graph
replay and eager execution (layer 0's output differs by 2e-3 with identical
kernel inputs and outputs; cause not determined).

The Kimi delta attention layer of Kimi Linear (`model.layers[i].self_attn` on
its delta attention layers) and the gated delta rule layer of OLMo hybrid
(`model.layers[i].linear_attn`) serve the same two sites. Both layers run every
request of a step that prefills through their chunk kernel. On a tiny random
Kimi Linear, every token's state of a 145-token prompt is within 0.7% of the
transformers sites in fp32. vLLM 0.33 cannot load an OLMo hybrid checkpoint:
loading `allenai`-named weights raises `TypeError: list indices must be
integers or slices, not NoneType` in the layer's conv weight loader, which
never receives the shard id. With that loader patched in a scratch run, a tiny
random OLMo hybrid was within 1.4% of the transformers sites at all 183
tokens; no test is committed for it.

The Kimi test also runs on an engine built with `taps`
(`"model.layers.*.self_attn.kernel"`) and on two tensor-parallel ranks, with
the same 3% bound against transformers. vLLM puts its graph break on the Kimi
layer's `_forward` itself, so the site is registered on the function under
that break, which is what a replayed step calls. The OLMo layer with `taps`
(`"model.layers.*.linear_attn.kernel"`) equalled the eager run exactly in the
scratch setup. On two ranks the tiny OLMo engine started and then hung on its
first trace, also with no site read; cause not determined.

Served on the Triton/FLA prefill kernel, which vLLM selects on GPUs before
Hopper, and on the one-token decode kernels. Not served: a prefill on the
FlashInfer kernel (vLLM's choice on Hopper), which returns no chunk states;
speculative decoding, where draft tokens advance states that are later
discarded; and the Kimi and OLMo gated delta rule layers.


## Mamba2 kernel sites

A Mamba2 layer (`MambaMixer2`: Mamba2, and the state-space layers of
Nemotron-H, Jamba, Granite hybrid, Falcon-H1 and Zamba2) carries the same two
recurrent sites:

```python
mixer = model.backbone.layers[0].mixer                        # mamba2-130m
with model.trace(prompt, max_tokens=1, temperature=0.0):
    chunks = mixer.kernel.chunk_states().clone().save()       # [chunks + 1, heads, head_dim, state]
    states = mixer.kernel.states(tokens=[0, 100, -1]).clone().save()
```

| Site | Shape | Value |
|------|-------|-------|
| `chunk_states` | `[chunks + 1, heads, head_dim, state]` | On a prefill step: the state entering each chunk the kernel cut for this request, then its final state. On a decode step: the state before the token and the state after it. |
| `states` | `[tokens, heads, head_dim, state]` | The state after each token of the step. A token that ends a chunk is the kernel's own state; one inside a chunk is advanced from the chunk's entering state in fp32 with the layer's recurrence. |

vLLM packs a step's prefilling requests into one strip and cuts chunks at every
multiple of the model's chunk size along the strip and at every request
boundary. A request that is not first in the strip therefore has a shorter
first chunk, and which request is first is vLLM's choice. `chunk_states`
follows that cut; `states` does not depend on it.

The state layout is the transformers layout, so no axes are swapped. On
mamba2-130m, the states after 8 tokens of a 391-token prompt equal the
transformers sites to 0.3% of the largest entry at layer 0 and 2.5% at layer 5,
with vLLM in bf16 and transformers in fp32. Under `tensor_parallel_size=2` the
sites assemble the ranks' heads and equal the one-rank engine to within 2%
(the test's bound).

On an engine built with `taps`, name the site as
`"backbone.layers.*.mixer.kernel"`. Layers 0 and 7 then equal the eager engine
exactly.

A Mamba1 layer (Mamba, Jamba) is served at the same location. Its prefill
kernel is one fused scan that writes only each request's final state, so a
request is one chunk: `chunk_states` is the state the request entered with and
its final state, and the other tokens of `states` are advanced from the
entering state in fp32 (`"derived"`). A Mamba1 state is `[channels, state]`
and `heads=` indexes the channels. On mamba-130m the states of four tokens of
a 73-token prompt are within 0.6% (layer 0) and 2% (layer 5) of the
transformers sites in fp32. The test runs on one rank, on two tensor-parallel
ranks and on an engine built with `taps`.

Not served: the ReplaySSM decode path and speculative decoding.

## Expert kernel sites

On a model whose MoE layers run vLLM's Triton experts kernel, the MoE module
(`MoERunner`, for example `layers[i].mlp.experts` on DeepSeek-V2) carries the
[expert kernel sites](../usage/kernel.md) of the transformers reader:

```python
mlp = model.model.layers[3].mlp                         # DeepSeek-V2-Lite
with model.trace(prompt, max_tokens=1, temperature=0.0):
    logits = mlp.gate.output[0].save()                               # [tokens, experts]
    act = mlp.experts.kernel.expert_activation().clone().save()      # [tokens, top_k, intermediate]
    out = mlp.experts.kernel.expert_output(tokens=-1).save()         # [top_k, hidden]
    one = mlp.experts.kernel.expert_output(expert=7).save()          # [tokens, hidden], 0 where not routed
```

| Site | Shape | Value |
|------|-------|-------|
| `expert_input` | `[tokens, top_k, hidden]` | The layer's input row. |
| `expert_activation` | `[tokens, top_k, width]` | The kernel's own activation rows, copied before vLLM reuses the workspace. |
| `expert_output` | `[tokens, top_k, hidden]` | The down projection before the routing weight. The kernel multiplies by the routing weight in the same pass and never stores this row, so the selected rows are recomputed from the activation rows and the expert's weight (provenance `replay`). |

There is no batch axis, and an invoke reads only its own tokens of the step.

Slot `k` is the token's `k`-th routed expert in the kernel's order. vLLM ranks
experts on bf16 scores, so for two experts with nearly equal scores the order
can differ from a top-k taken over `gate.output` in fp32. `expert=` selects by
expert id and does not depend on the order.

The MoE module's `.output` is the weighted sum of the routed rows plus the
shared experts when the model has them. On DeepSeek-V2-Lite layer 3, the
activation and output rows equal an fp32 computation from the layer's weights
to 1% of each row's largest entry, and the weighted sum plus the shared experts
equals the module output to 0.7% per token.

Under expert parallelism (`tensor_parallel_size=2, enable_expert_parallel=True`)
each rank holds a share of the experts and runs only the pairs routed to them.
The sites sum the ranks, so a read returns every routed pair's row, as on one
rank. On DeepSeek-V2-Lite layer 3 with two ranks, the three sites equal the
one-rank engine to 2% of the largest entry, which is also how much the layer's
input differs between the two engines.

Under tensor parallelism without expert parallelism each rank holds every
expert, cut to one block of its intermediate width. The activation rows are
joined along the width, and `expert_output` is the sum of the ranks' partial
down projections. On DeepSeek-V2-Lite layer 3 with two ranks the three sites
equal the one-rank engine to within 5% of the largest entry (the test's bound).

`expert_activation` also takes `width=` to select intermediate units, on one
rank and across ranks: `kernel.expert_activation(tokens=-1, width=[3, 700])`.

On an engine built with `taps`, name the site as
`"model.layers.*.mlp.experts.kernel"`. vLLM records the MoE layer into the CUDA
graph whole, so a replayed step would run no Python there. For a MoE module
with a kernel tap, nnsight keeps the module's body in Python as a graph break
of its own and copies its output into a buffer the next graph segment reads.
Module locations inside that body (the gate, for example) are then served on
every step when they are taps. Layer 1 equals the eager engine; at layer 7 the
rows differ by a few percent, and one pair of 30 was routed to another expert,
because a replayed graph rounds differently from eager execution upstream.

Not served: other experts kernels (FlashInfer, batched Triton, quantized
weights or activations) and LoRA.

## CUDA graphs with taps

vLLM's decode throughput comes largely from CUDA-graph replay, which `enforce_eager=True` gives up because a replayed graph runs no Python and so no hooks. `taps` keeps the graphs: the locations you name are recorded *into* the graph as breaks (vLLM's `VLLM_USE_BREAKABLE_CUDAGRAPH`), and at each break the interleaver's normal handoff runs on every replay. Everything else is unchanged — the same trace syntax, per-request scoping, `tracer.iter`, `model.logits` / `model.samples`.

```python
model = VLLM(
    "meta-llama/Llama-3.1-8B",
    dispatch=True,
    taps=["model.layers.*.output", "model.layers.10.mlp.input"],
)

with model.trace(prompt, max_tokens=20, temperature=0.0) as tracer:
    hiddens = list().save()
    for _ in tracer.iter[:20]:                                     # every step, prefill included
        model.model.layers[10].output[0][:] += 4 * steering_vector   # in place
        hiddens.append(model.model.layers[16].output[0].clone())     # clone: see below
```

Outside the loop the edit would fire on the prefill only.

What changes under graphs, and why:

- **A tap can be an operation inside a forward.** `"model.layers.10.self_attn.source.qkv_split_0.output"` taps a `.source` op: the worker installs the source-instrumented forward for exactly the modules such taps name, before the graphs are recorded, and the op is served on replay like any module location (reads bitwise-equal to the eager engine, in-place edits land). An op name the forward does not have is refused while the engine builds; the caller gets
  `RuntimeError: Engine core initialization failed`, and the message that names the ops the forward
  *does* have is in the EngineCore subprocess's output above it (see [When the engine builds, or
  does not](#when-the-engine-builds-or-does-not)). Ops inside a fused kernel are still not locations.
- **Only taps are reachable.** A location you didn't declare is never visited by a replayed step, so a block that reads one parks until the request ends and the request's error says `'...' is not a tap on this engine`. Declare it, or drop `taps` for the eager engine. `model.taps` lists the resolved set after dispatch. Keep the set small: each tap splits the graph, and a break at every module would cost what replay bought.
- **Edits land in place.** The next kernel reads the tap's tensor from a fixed address, so an in-place edit (`output[0][:] += v`, `output[0][:, i] = 0`) is exactly right and a replacement (`output = t`) is copied back into that memory, so it has to keep the rows it
  replaces — see [Replacing a value](#replacing-a-value-keeps-its-rows).
- **Clone what you keep — and nothing warns.** The value served at a tap *is* the graph's memory, rewritten next step. Un-cloned decode entries appended under `tracer.iter` all come back holding the last step's values (the prefill entry is a different buffer and survives). See [Serving copies instead of views](#serving-copies-instead-of-views).
- **`torch.compile` is off.** Breakable graphs keep replay and drop the compiled path; that is most of the throughput, not all of it. The flag is process-wide, so one process holds either graph engines or compiled ones.
- **The graphs are piecewise.** A tapped engine sets `cudagraph_mode="PIECEWISE"`, since a tap is a break in a piecewise graph and a full graph has none. A `compilation_config` of your own that asks for full graphs is refused while the engine builds.

## Serving copies instead of views

What a block reads is a view into the engine's own memory, on the eager engine and at a tap
alike. That is what makes an in-place edit work — the block writes where the model will read next —
and it is why a value kept past the point it was read is unreliable: fused kernels
(`fused_add_rms_norm`, MLA's in-place rotation of the `q_proj` output) overwrite those buffers a few
ops later, so `layers[8].output[0].save()` comes back holding a later layer's data. The per-site fix
is `.clone()` or a reduction before `.save()` (`(out[0] + out[1]).mean(0).cpu().save()`), which is
easy to forget and silent when forgotten.

`NNSIGHT_VLLM_CLONE_READS=1` makes the trade once, for a whole engine: every value handed to a block
is a private copy of the request's own rows, so `.save()`, `tracer.cache()` and appends under
`tracer.iter` keep what was computed. Set it in the environment before `VLLM(...)` builds the engine
(the copies are made in the worker process), not in `CONFIG`. It costs one allocation per module
read per step, and **in-place edits do not survive it** — they write to the copy. Assign the edited
value back instead, which lands under either setting:

```python
out = model.model.layers[10].output
out[0][:] += v                                    # alone, dropped under CLONE_READS
model.model.layers[10].output = out               # lands under either setting
```

## Where an error comes from

Your block runs in vLLM's EngineCore subprocess, and that decides where you read about a failure.

**Inside a trace, errors come home.** An exception in the block is re-raised in your process as a
`RuntimeError` carrying the original type, its message and an "Intervention traceback" pointing at
your line — `1/0` in a block arrives as `RuntimeError: ZeroDivisionError: division by zero`.
Because the type is carried in the message rather than in the class, catch `RuntimeError` and match
on the text. A trace that raises binds none of its saved names, not even another invoke's. The
engine keeps serving afterwards, and requests from other clients in the same batch are unaffected.

**Warnings do not.** A warning raised while the block runs prints to the EngineCore process's
output, prefixed `(EngineCore pid=...)`, and a `warnings.catch_warnings()` around the trace
records nothing.

### When the engine builds, or does not

Everything a `VLLM(...)` constructor does in the worker fails the same shape:

```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

"Above" is literal. The real message is in the `(EngineCore pid=...)` lines further up the output,
usually many screens of vLLM logging back. Two causes are common enough to recognize:

- a `.source` op named in `taps=` that the forward does not have. (A tap naming no *module* is
  caught client-side, against the meta tree, and raises `ValueError` where you wrote it; the op is
  checked in the worker, where the forward is instrumented, and the EngineCore line names the ops
  it *does* have.)
- `AssertionError: Error in memory profiling. Initial free memory ... current free memory ...` on a
  shared GPU, when another process frees memory while vLLM profiles. Nothing is wrong with the
  trace: build again, or use a card nothing else is holding.

`Chunked prefill is enabled with max_num_batched_tokens=...` is printed on every construction and
does not describe the engine you get. It comes from the meta tree nnsight builds first; the real
engine's arguments are logged a few lines later as `non-default args: {... 'enable_chunked_prefill':
False ...}`.

## Limitations

- **One prompt per invoke** — no `tracer.invoke(["a", "b"])`.
- **`.skip()` is effectively unsupported** — see [`.skip()` needs the whole step](#skip-needs-the-whole-step).
- **No `tracer.barrier(n)`** — each invoke is its own request and the engine schedules them independently, so the blocks never run against the same forward. Calling it raises rather than hanging.
- **No speculative decoding** — a speculative step scores several draft tokens per request, and a block is scoped to one logits row and one sampled token per request. `VLLM(..., speculative_config=...)` raises `NotImplementedError` before anything is built.
- **No batch-sharded sampling** — each tensor-parallel rank then samples only the requests it owns, and a block's logits are scoped to the whole batch. `VLLM(..., enable_batch_sharded_sampling=True)` raises `NotImplementedError` before anything is built.
- **No data parallelism** — vLLM answers the call that brings a request's saved values back from its first engine only, so `VLLM(..., data_parallel_size=N)` with `N > 1` raises `NotImplementedError` before anything is built. Use `tensor_parallel_size`, or one `VLLM` per GPU.
- **No backward / gradients, and no source tracing inside a fused CUDA kernel** — the kernel's
  inputs and outputs are locations; its interior is not Python. Of the [kernel sites](../usage/kernel.md),
  only attention on the FlashAttention backend is served ([Attention kernel sites](#attention-kernel-sites)).
  Recurrent and MoE sites such as `mixer.kernel.states(...)` are unavailable: vLLM's Mamba, gated-delta
  and fused-MoE layers execute through vLLM's own custom ops and state caches, so requesting `.kernel`
  there ends in `OutOfOrderError` because the location is never reached.
- **No `.scan()`.** It propagates shapes by running the model's own forward under a fake-tensor
  mode, and there is no forward on this side to run. `model.scan(...)` raises
  `NotImplementedError: scan is unavailable on vLLM: ... Trace a prompt and read the shapes off the
  activations it serves.`
- **Text prompts only** — image/video inputs are not accepted; vision-language checkpoints load and their language trunk traces normally.
- **A model without a native vLLM definition traces, with one exception.** vLLM serves it through
  its Transformers backend, which runs the wrapped HuggingFace module with a leading singleton
  batch dim, so its activations are `[1, total_tokens, hidden]` rather than `[total_tokens, hidden]`.
  Reads and writes are scoped to the right tokens either way. Declaring `taps` on such a model is
  not covered: the graph-replay path still trims padding on dim 0, so a tap can be served the
  step's padding rows.

## vLLM versions

nnsight is built and tested against vLLM **0.33**, and the `vllm` extra pins
`vllm>=0.33,<0.34`. nnsight subclasses vLLM's GPU model runner and imports its internals directly
(`vllm.tokenizers`, `vllm.v1.worker.gpu.model_runner`, `vllm.v1.engine.async_llm`), which move between
minor releases. A vLLM installed outside the pin is untested and can be wrong without an error.

vLLM 0.33 still carries its older model runner and falls back to it under `VLLM_USE_V2_MODEL_RUNNER=0`,
or for an option only that runner supports (its log names which). nnsight does not instrument it, so
such an engine is refused while it builds.

## Gotchas

- **Scripts need an `if __name__ == "__main__":` guard.** Dispatching an engine initializes CUDA in your process, so vLLM forces the `spawn` start method for its workers (logging `We must use the spawn multiprocessing start method ... Reasons: CUDA is initialized`). Spawning re-imports the main module; without the guard the child re-runs your top-level `VLLM(...)` / `.trace(...)` and the engine core dies with `An attempt has been made to start a new process before the current process has finished its bootstrapping phase`. This holds on one card and `mode="sync"` as much as under tensor parallelism. Notebooks are fine. It is a vLLM/multiprocessing requirement, not nnsight-specific.

  ```python
  from nnsight.modeling.vllm import VLLM

  def main():
      model = VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True)
      with model.trace("hello", max_tokens=1):
          out = model.transformer.h[0].output.save()
      print(out.shape)

  if __name__ == "__main__":
      main()
  ```

- **A saved value comes back by its variable *name* — bind it.** `logits = model.logits.save()` works; a bare `model.logits.save()` marks the value but has no name to return it under, so `output.saves["logits"]` (async/serve) or the pushed-back local is silently missing. This is the most common "saves don't come back" bug on vLLM.
- **An empty `tracer.invoke()` with interventions raises** (its work would vanish); a do-nothing empty invoke is a harmless no-op.

## Related

- [docs/models/vllm-editing.md](vllm-editing.md) — `model.edit()` on the engine
- [docs/models/vllm-serving.md](vllm-serving.md) — async mode, `nnsight-serve`, remote
- [docs/models/vllm-parallelism.md](vllm-parallelism.md) — tensor parallelism, MoE, hybrid trunks
- [docs/models/index.md](index.md) — decision tree
- [docs/models/transformers-model.md](transformers-model.md) — the text-only HF alternative
- [docs/developing/vllm-integration.md](../developing/vllm-integration.md) — how the worker, batcher and collect path work
- `src/nnsight/modeling/vllm/` — source
- `tests/vllm/` — runnable examples (tracing, async, tensor parallelism, mixture-of-experts, requests, serve)