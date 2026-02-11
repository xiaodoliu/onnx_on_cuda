import argparse
import time
import onnxruntime_genai as og

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model_dir", required=True)
    p.add_argument("--prompt", default="Tell me a joke about GPUs.")
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    args = p.parse_args()

    # ORT GenAI 0.11.x: use Config to choose execution provider
    cfg = og.Config(args.model_dir)
    cfg.clear_providers()
    cfg.append_provider("cuda")   # <-- use CUDA on your RTX 4090

    model = og.Model(cfg)
    tok = og.Tokenizer(model)

    def run_once():
        params = og.GeneratorParams(model)
        # max_length means total sequence length; for simple perf we just cap output length
        params.set_search_options(max_length=args.max_tokens)

        gen = og.Generator(model, params)
        gen.append_tokens(tok.encode(args.prompt))

        t0 = time.perf_counter()
        n = 0
        while not gen.is_done():
            gen.generate_next_token()
            n += 1
        t1 = time.perf_counter()

        text = tok.decode(gen.get_sequence(0))
        return n, (t1 - t0), text

    # warmup
    for _ in range(args.warmup):
        run_once()

    # timed runs
    times = []
    tokens = []
    last_text = ""
    for _ in range(args.iters):
        n, dt, last_text = run_once()
        times.append(dt)
        tokens.append(n)

    avg_tps = sum(tokens) / sum(times)
    print(f"Runs: {args.iters}, tokens generated per run: {tokens}")
    print(f"Times (s): {[round(x,4) for x in times]}")
    print(f"Avg tokens/sec: {avg_tps:.2f}")
    print("\n--- sample output ---\n")
    print(last_text)

if __name__ == "__main__":
    main()
