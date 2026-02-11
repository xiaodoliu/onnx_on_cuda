import argparse
import time
from pathlib import Path


def main() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency 'numpy'. Install it in your venv:\n"
            "  python -m pip install numpy\n"
        ) from e

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--disable_optimizations", action="store_true")
    p.add_argument(
        "--depth",
        type=int,
        default=0,
        help="override model depth; 0 means read from ONNX metadata (bench.depth) or assume 1",
    )
    p.add_argument(
        "--count_ops",
        type=float,
        default=1.0,
        help="ops per element per layer. add/mul/exp/sin=1; fma(mul+add)=2. This is only an estimate for complex blocks.",
    )
    args = p.parse_args()

    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("bad --warmup/--iters")

    import onnxruntime as ort

    # Determine model depth from metadata written by the generator.
    depth = 1
    if args.depth and args.depth > 0:
        depth = args.depth
    else:
        try:
            import onnx

            m = onnx.load(str(args.model))
            meta = {p.key: p.value for p in m.metadata_props}
            if "bench.depth" in meta:
                depth = max(1, int(meta["bench.depth"]))
        except Exception:
            depth = 1

    so = ort.SessionOptions()
    if args.disable_optimizations:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    sess = ort.InferenceSession(
        str(args.model),
        sess_options=so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    if inp.type != "tensor(float16)":
        raise SystemExit(f"Expected fp16 input, got {inp.type}")

    # Static shapes are expected (generator uses static dims).
    shape = [int(d) for d in inp.shape]
    if any(d <= 0 for d in shape):
        raise SystemExit(f"Model input shape not static: {inp.shape}")

    x = (np.random.rand(*shape).astype(np.float16) * np.float16(0.5)) + np.float16(0.01)

    # Warmup.
    for _ in range(args.warmup):
        sess.run([out.name], {inp.name: x})

    # Timed.
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run([out.name], {inp.name: x})
        t1 = time.perf_counter()
        times.append(t1 - t0)

    best = min(times)
    avg = sum(times) / len(times)
    elems = int(np.prod(shape))

    # This is "effective ops" based on your count and model depth.
    # For fma as (mul+add), set count_ops=2. For complex blocks (softmax/norm/gelu),
    # compare latency and kernel mix first; this number is only a rough proxy.
    total_ops = elems * float(depth) * float(args.count_ops)
    best_gops = (total_ops / best) / 1e9
    avg_gops = (total_ops / avg) / 1e9

    print("Providers:", sess.get_providers())
    print(f"model={args.model}")
    print(f"input={inp.name} shape={shape} dtype=fp16 elems={elems}")
    print(f"output={out.name} dtype={out.type}")
    print(f"warmup={args.warmup} iters={args.iters} disable_optimizations={args.disable_optimizations}")
    print(f"depth={depth} count_ops(per_element_per_layer)={args.count_ops}")
    print(f"best_s={best:.6f} avg_s={avg:.6f}")
    print(f"effective_throughput(best)={best_gops:.3f} GOP/s  effective_throughput(avg)={avg_gops:.3f} GOP/s")


if __name__ == "__main__":
    main()

