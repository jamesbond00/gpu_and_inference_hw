# HW2 Implementation Notes

This note explains the HW2 optimization plan and gives a practical GPU run
checklist.

## Goal

HW2 starts from a deliberately slow autoregressive generation loop and asks us
to make it faster.

The model is a tiny random Llama. It is not trained and the generated text does
not matter. The assignment is about inference performance:

- avoid recomputing work
- avoid CPU/GPU synchronization inside the token loop
- avoid repeated tensor allocation and copying
- use an inference-friendly dtype
- produce profiler traces that explain the speedup

The score is based on speedup against the provided `slow_loop` in `utils.py`.

## Baseline Problem

The slow baseline repeatedly feeds the whole growing sequence back into the
model:

```python
generated_ids = input_ids.clone()
generated_tokens = []
for _ in range(n_steps):
    outputs = model(input_ids=generated_ids)
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    token_value = next_token_id.item()
    generated_tokens.append(token_value)
    generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
```

This has several intentional performance problems.

### Full-Sequence Recomputation

At decode step 1, the model sees the original prompt.

At decode step 2, it sees prompt plus one generated token.

At decode step 3, it sees prompt plus two generated tokens.

That means the same prompt tokens are recomputed again and again. For a
transformer, this is expensive because attention and hidden states are rebuilt
for old tokens that have not changed.

The correct inference pattern is:

1. Run the full prompt once.
2. Save the model's KV cache.
3. For each new token, run only that one token while reusing the cache.

This is expected to be the largest speedup.

### `.item()` Synchronization

This line is also expensive:

```python
token_value = next_token_id.item()
```

GPU execution is asynchronous. Python can normally launch CUDA work and continue.
But `.item()` asks for a Python scalar, so the CPU has to wait until the GPU has
finished producing that value.

Doing this once per generated token creates a CPU/GPU synchronization point
inside the hottest loop.

The optimized loop should keep tokens as CUDA tensors during generation and only
copy them back to CPU at the end.

### Repeated `torch.cat`

This line reallocates and copies the growing sequence every step:

```python
generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
```

With KV caching, the loop does not need to keep feeding the full growing
sequence. It only needs the latest token and the cache returned by the model.

### Float32 Baseline

The baseline model is built with:

```python
build_model(torch.float32)
```

For inference on NVIDIA GPUs, lower precision is usually much faster. On L40S,
`torch.float16` is a good first choice for this homework because model quality is
irrelevant and the task only needs the loop to run.

## Functions to Implement

All HW2 edits should stay in `hw2_task.py`.

### `profile(loop_fn, model, input_ids, trace_name)`

This function should wrap:

```python
loop_fn(model, input_ids, PROFILE_STEPS)
```

with `torch.profiler`.

It should:

- record CPU activity
- record CUDA activity
- print a profiler summary table
- export a Chrome trace to `RESULTS_DIR / trace_name`

Expected trace files:

```text
hw2/results/v0_slow_trace.json
hw2/results/v1_optimized_trace.json
```

These traces can be opened in:

```text
https://ui.perfetto.dev
```

The trace is mainly for understanding what changed. The final grade speedup
comes from the unprofiled timing helper `time_generation`.

### `optimized_loop(model, input_ids, n_steps)`

The optimized loop should use the Llama KV cache.

High-level structure:

```python
with torch.inference_mode():
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

    generated = [next_token]

    for _ in range(n_steps - 1):
        outputs = model(
            input_ids=next_token[:, None],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated.append(next_token)

    return torch.cat(generated).detach().cpu().tolist()
```

The important details are:

- the prompt is processed once
- each decode step processes one token
- `past_key_values` is passed back into the model
- no `.item()` is called inside the loop
- no growing `torch.cat` is used as model input
- the return value is converted to a Python list at the end

The returned list keeps compatibility with `time_generation`, which prints a
token preview.

### `generate_optimized(optimized_trace_name)`

This function should:

1. Build the optimized model.
2. Get input IDs.
3. Run `profile`.
4. Run `time_generation`.
5. Return the elapsed time.

Planned model dtype:

```python
model = build_model(torch.float16)
```

The baseline still uses `torch.float32` because `main()` creates it that way.
Only the optimized model should change dtype.

## Why KV Cache Is the Main Optimization

Autoregressive generation predicts one token at a time. Previous tokens do not
change, so their key/value attention states do not need to be recomputed.

The KV cache stores those states.

Without cache:

```text
step 1: process 1024 tokens
step 2: process 1025 tokens
step 3: process 1026 tokens
...
```

With cache:

```text
prefill: process 1024 tokens once
step 1: process 1 token
step 2: process 1 token
step 3: process 1 token
...
```

That changes the shape of the work completely. It should be the largest single
speedup in this assignment.

## Expected Trace Differences

In the slow trace, expect:

- repeated full-sequence model forwards
- increasing sequence lengths over time
- CPU/GPU sync points from `.item()`
- repeated allocation/copy behavior from `torch.cat`

In the optimized trace, expect:

- one larger prefill call
- many smaller one-token decode calls
- fewer CPU/GPU sync points
- less repeated sequence copying

The optimized trace may still show many operations because the model is still a
PyTorch transformer, but the structure should be clearly better.

## Running on the GPU VM

From the repository root:

```bash
source .venv/bin/activate
python3 hw2/hw2_task.py
```

If setting up from scratch:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check GPU visibility:

```bash
python3 - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
PY
```

Expected for our VM:

```text
CUDA: True
GPU: NVIDIA L40S
torch: 2.9.1+cu128
```

## Expected Output

The script should print:

```text
--- Part 1: Slow baseline ---
Slow: 128 tokens in ...

--- Part 2: Optimized ---
Optimized: 128 tokens in ...

SUMMARY
  Slow:       ...
  Optimized: ...
  Speedup:   ...x
```

The target is:

```text
Speedup >= 4.0x
```

The minimum useful target is:

```text
Speedup >= 3.0x
```

## Output Files to Save

After the run, save:

```text
hw2/hw2_task.py
hw2/results/v0_slow_trace.json
hw2/results/v1_optimized_trace.json
```

Also save the terminal output containing the final timing summary.

## If Speedup Is Too Low

Check these in order:

1. Confirm the optimized loop uses `past_key_values`.
2. Confirm the model is built with `torch.float16`.
3. Confirm there is no `.item()` inside the generation loop.
4. Confirm the loop is not concatenating and feeding the full generated sequence.
5. Open both traces in Perfetto and look for repeated long full-sequence forwards
   in the optimized trace.

Possible further experiments:

- try `torch.bfloat16`
- try `torch.backends.cuda.matmul.allow_tf32 = True` for FP32 experiments
- use model call options that reduce logits work if supported by the installed
  Transformers version

Keep changes inside `hw2_task.py` unless the assignment explicitly allows
otherwise. `utils.py` should remain unchanged.
