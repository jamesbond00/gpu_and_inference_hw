import torch
from torch.profiler import ProfilerActivity, profile

from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    """Generate tokens using KV cache instead of recomputing the whole prefix.

    The slow baseline feeds the entire growing sequence back to the model on
    every step. Here we prefill the prompt once, then decode one new token at a
    time while passing `past_key_values` back into the model.
    """
    if n_steps <= 0:
        return []

    generated_tokens = []

    with torch.inference_mode():
        # Prefill: process the full prompt once and keep the KV cache.
        #
        # logits_to_keep=1 asks Transformers to compute logits only for the
        # final position. For generation we only need the next-token logits, not
        # a vocab projection for every prompt token.
        outputs = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id)

        # Decode: each remaining step consumes only the latest token.
        for _ in range(n_steps - 1):
            outputs = model(
                input_ids=next_token_id[:, None],
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token_id)

        # Convert to Python integers once at the end. Calling .item() inside the
        # loop would force a CPU/GPU synchronization for every generated token.
        return torch.cat(generated_tokens, dim=0).detach().cpu().tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    """Run a short profiled generation and save a Chrome trace."""
    trace_path = RESULTS_DIR / trace_name
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)
        torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )
    )
    prof.export_chrome_trace(str(trace_path))
    print(f"Chrome trace saved to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    """Build the optimized model, profile it, time it, and return elapsed time."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = build_model(torch.float16)
    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    return time_generation(optimized_loop, model, input_ids, "Optimized")


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
# - Used the model KV cache: one full-prompt prefill followed by one-token
#   decode steps, instead of recomputing the whole growing sequence every step.
# - Moved token conversion to the end of the loop, avoiding per-token `.item()`
#   CPU/GPU synchronizations.
# - Removed repeated `torch.cat` on the growing input sequence; decode now feeds
#   only the latest token plus `past_key_values`.
# - Built the optimized model in float16, which is a better inference dtype for
#   the L40S than the float32 baseline.
# - Used `logits_to_keep=1` so the model only computes next-token logits needed
#   for generation.
#
# Fill in measured timings after the GPU run:
# - Slow baseline:
# - Optimized:
# - Speedup:
#
# Biggest impact and why:
# The KV cache should be the biggest win. It changes decode from repeated
# full-sequence forward passes into one prompt prefill plus cheap one-token
# forward passes.
#
