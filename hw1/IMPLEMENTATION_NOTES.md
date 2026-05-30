# HW1 Implementation Notes

This note explains the implementation in `hw1_task_impl.py` and gives a
practical checklist for running the benchmark on a GPU VM.

## Goal

HW1 is about the GPU roofline model. The roofline model compares:

- how much data a kernel moves, measured in bytes
- how much math it performs, measured in FLOPs
- how fast the GPU can move memory
- how fast the GPU can do floating-point math

The key metric is arithmetic intensity:

```text
arithmetic intensity = FLOPs / bytes moved
```

Low arithmetic intensity usually means the kernel is memory-bound. It spends
most of its time waiting for data movement.

High arithmetic intensity usually means the kernel is compute-bound. It has
enough math per byte that arithmetic throughput becomes the limiter.

## Implemented Functions

### `lowest_ai_fn`

```python
return x.clone()
```

This is used as the lowest-arithmetic-intensity baseline.

`clone()` copies the tensor on the GPU. It reads the input tensor and writes a
new output tensor, but it does no useful floating-point arithmetic. That makes
it a good practical approximation of "mostly memory traffic, almost zero FLOPs."

The runtime code treats this point as a tiny nonzero AI value so it can appear
on the log-scale roofline plot.

### `make_compute_fn`

The generated function does:

```python
acc = x
for _ in range(num_ops):
    acc = acc * x + x
return acc
```

Each loop iteration performs, per element:

- one multiply
- one add

So each iteration is counted as 2 FLOPs per element.

The tensor shape stays the same no matter how large `num_ops` is. That is
important: increasing `num_ops` increases the amount of math without changing
the input/output tensor size.

The function can be returned in two modes:

```python
return torch.compile(fn) if compiled else fn
```

This lets the benchmark compare eager PyTorch against a compiled version.

### Why Compiled and Eager Behave Differently

In compiled mode, `torch.compile` can fuse this simple element-wise chain into a
small number of kernels, often one kernel. In the ideal fused model:

- read `x` once
- keep intermediate `acc` values in registers
- write the final result once

So the byte traffic at the kernel boundary is modeled as:

```text
bytes = num_elements * bytes_per_element * 2
```

The `2` means one read and one write.

In eager mode, PyTorch usually launches separate kernels for the multiply and
add in every loop iteration. Intermediates are materialized in global memory.

For each loop iteration, the byte model is:

```text
multiply: read acc + read x + write tmp = 3 element transfers
add:      read tmp + read x + write acc = 3 element transfers
```

So eager traffic is modeled as:

```text
bytes = num_elements * bytes_per_element * num_ops * 6
```

This is why the compiled points should move rightward on the roofline plot as
`num_ops` increases, while eager points stay much more memory-traffic-heavy.

### `benchmark_fn`

The benchmark uses CUDA events:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
fn(*args)
end.record()
end.synchronize()
elapsed_ms = start.elapsed_time(end)
```

CUDA events measure elapsed GPU time between two points in the CUDA stream. This
is better than plain Python wall-clock timing for kernels because GPU work is
asynchronous: Python can launch work and continue before the GPU has finished.

The function also:

- runs warmup iterations first
- synchronizes after warmup
- records many repetitions
- returns the median latency

The median is used because occasional outliers are common in GPU timing.

### `compute_elementwise_metrics`

The function returns:

```python
total_flops, arithmetic_intensity, achieved_flops
```

The FLOP count is:

```text
total_flops = num_elements * num_ops * 2
```

The achieved FLOP/s is:

```text
achieved_flops = total_flops / seconds
```

where:

```text
seconds = ms * 1e-3
```

The arithmetic intensity is:

```text
ai = total_flops / total_bytes
```

`total_bytes` depends on whether the variant is `compiled` or `eager`, as
explained above.

## Running on a GPU VM

Run all commands from the repository root:

```bash
cd /path/to/gpu_and_inference_hw
```

Create a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that PyTorch can see the GPU:

```bash
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
PY
```

Expected result:

```text
CUDA available: True
GPU: ...
```

Then run HW1:

```bash
python3 hw1/hw1_task.py
```

The first run may take longer because `torch.compile` needs to compile kernels.

## Expected Output Files

After a successful run, check:

```bash
ls -lh hw1/results/
```

Expected files:

```text
roofline.png
roofline_data.json
```

`roofline.png` is the main plot for the assignment.

`roofline_data.json` contains the raw measurements.

## Sanity Checks

The run should print rows like:

```text
lowest-AI: ...
1 ops (eager): ...
1 ops (compiled): ...
...
128 ops (compiled): ...
matmul 1024x1024: ...
```

Important qualitative checks:

- The script completes without errors.
- The plot points are below the theoretical roofline ceiling.
- The compiled `ops-K` points move right as `K` increases.
- The eager `ops-K` points do not move right in the same way.
- `roofline.png` is created and visually readable.

If the GPU is not H100 or L40S, `hw1_runtime.py` may raise an unsupported GPU
error. In that case, add the GPU's FP32 peak FLOP/s and memory bandwidth to
`GPU_SPECS` in `hw1_runtime.py`.

## What to Save Before the VM Is Erased

Copy these back to your local machine or commit them before the GPU slot ends:

```text
hw1/hw1_task_impl.py
hw1/IMPLEMENTATION_NOTES.md
hw1/results/roofline.png
hw1/results/roofline_data.json
```

Also save the terminal output if the checker asks for measured numbers.
