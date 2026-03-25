import argparse
import time

from nanovllm import LLM, SpeculativeLLM, SamplingParams


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke/benchmark test for Speculative Decoding")
    parser.add_argument("--draft-model", type=str, required=True, help="Path to draft model (e.g. Qwen3-0.6B)")
    parser.add_argument("--target-model", type=str, required=True, help="Path to target model (e.g. Qwen3-4B)")
    parser.add_argument("--k", type=int, default=1, help="Speculative window size")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--no-baseline", action="store_true", help="Skip target-only baseline")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Can be used multiple times. If omitted, use built-in prompts.",
    )
    return parser


def default_prompts():
    return [
        "Explain speculative decoding in simple terms.",
        "Give me a 5-step plan to optimize LLM inference latency.",
    ]


def main():
    args = build_parser().parse_args()
    prompts = args.prompt if args.prompt else default_prompts()

    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )

    common_kwargs = dict(
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )

    print("=" * 80)
    print("[Speculative] Initializing engines...")
    spec = SpeculativeLLM(
        draft_model=args.draft_model,
        target_model=args.target_model,
        k=args.k,
        **common_kwargs,
    )

    t0 = time.perf_counter()
    spec_outputs = spec.generate(prompts, sp, use_tqdm=True)
    spec_time = time.perf_counter() - t0

    print("\n[Speculative] Done")
    print(f"Elapsed: {spec_time:.3f}s")
    for i, out in enumerate(spec_outputs):
        print(f"- Prompt {i}: {len(out['token_ids'])} tokens")

    baseline_time = None
    baseline_outputs = None

    if not args.no_baseline:
        print("\n" + "=" * 80)
        print("[Baseline] Initializing target-only LLM...")
        base = LLM(args.target_model, **common_kwargs)
        t1 = time.perf_counter()
        baseline_outputs = base.generate(prompts, sp, use_tqdm=True)
        baseline_time = time.perf_counter() - t1
        base.exit()

        print("\n[Baseline] Done")
        print(f"Elapsed: {baseline_time:.3f}s")
        for i, out in enumerate(baseline_outputs):
            print(f"- Prompt {i}: {len(out['token_ids'])} tokens")

    spec.exit()

    print("\n" + "=" * 80)
    print("Summary")
    print(f"Speculative(k={args.k}): {spec_time:.3f}s")
    if baseline_time is not None:
        speedup = baseline_time / max(spec_time, 1e-9)
        print(f"Baseline(target-only): {baseline_time:.3f}s")
        print(f"Speedup: {speedup:.3f}x")

    print("\nSample output (Speculative):")
    for i, out in enumerate(spec_outputs):
        text = out["text"].replace("\n", " ")
        print(f"[{i}] {text[:200]}{'...' if len(text) > 200 else ''}")


if __name__ == "__main__":
    main()
