import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # clone() performs a device-to-device copy: it reads x and writes the result
    # without doing useful floating-point math. That makes it a good practical
    # baseline for the memory-bandwidth side of the roofline.
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        # Keep the input/output shape fixed so increasing num_ops changes the
        # amount of arithmetic, not the amount of boundary memory traffic.
        # Each iteration contributes one multiply and one add per element.
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))

    return float(torch.tensor(timings_ms).median().item())


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # The loop body is acc = acc * x + x.
    # Per element, per loop iteration:
    #   1 multiply + 1 add = 2 FLOPs.
    total_flops = num_elements * num_ops * 2

    if variant == "compiled":
        # torch.compile can fuse the whole pointwise chain into one kernel for
        # this simple static function. In the ideal fused model, x is read once
        # and the final acc is written once; intermediate acc values stay inside
        # registers instead of going back to global memory after every op.
        total_bytes = num_elements * bytes_per_element * 2
    elif variant == "eager":
        # Eager PyTorch launches separate pointwise kernels for the multiply and
        # the add in each Python-loop iteration. Approximate global-memory
        # traffic per iteration as:
        #   multiply: read acc + read x + write tmp = 3 element transfers
        #   add:      read tmp + read x + write acc = 3 element transfers
        # That is 6 element transfers per iteration.
        total_bytes = num_elements * bytes_per_element * num_ops * 6
    else:
        raise ValueError(f"unknown elementwise variant: {variant!r}")

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
# A1. In the compiled case, the pointwise chain is fused. The kernel still reads
# the input once and writes the output once, so the boundary memory traffic is
# roughly constant as K grows. Adding more FMA-style loop iterations increases
# FLOPs much faster than bytes. Runtime can remain close to the memory-copy time
# for small and medium K, but the numerator in FLOP/s grows, so achieved FLOP/s
# rises and the points move right/up on the roofline.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
# A2. A 1024x1024 FP32 matmul is not always large enough to fully occupy a big
# GPU, so fixed overheads, launch latency, and imperfect SM occupancy can matter.
# Library matmul also has tiling and scheduling overhead, while the compiled
# element-wise kernel is a very simple, regular stream of independent FMAs over a
# huge vector. For small matrix sizes, that simple kernel can report higher
# non-tensor-core FP32 utilization than the small GEMM.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
# A3. It suggests the kernel is moving away from the memory-bandwidth-limited
# region and toward the compute-limited region. Once there is enough arithmetic
# per byte, extra loop iterations can no longer be hidden behind the same memory
# traffic, so additional math starts increasing runtime.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
# A4. Eager mode executes the Python loop as many separate PyTorch operations.
# Each multiply and add launches its own kernel and materializes intermediates in
# global memory, so bytes moved grow with K. The compiled version can fuse the
# chain and keep intermediates in registers, so bytes stay close to one read and
# one write while FLOPs grow with K. That is why compiled points move to higher
# arithmetic intensity, while eager points stay much more memory-traffic-heavy.
