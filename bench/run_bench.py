import argparse
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_CU = REPO_ROOT / "bench" / "bench_libdevice.cu"
OUT_BIN = REPO_ROOT / "bench" / "bench_libdevice"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def build(nvcc: str, arch: str, use_fast_math: bool) -> None:
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nvcc,
        "-O3",
        "-lineinfo",
        f"-arch={arch}",
        str(BENCH_CU),
        "-o",
        str(OUT_BIN),
    ]
    if use_fast_math:
        cmd.insert(1, "--use_fast_math")
    run(cmd, cwd=REPO_ROOT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    p.add_argument(
        "--arch",
        default=os.environ.get("CUDA_ARCH", "sm_89"),
        help="e.g. sm_89 for RTX 4090",
    )
    p.add_argument("--use_fast_math", action="store_true")
    p.add_argument("--no_build", action="store_true")

    # pass-through for the benchmark binary
    p.add_argument("--op", default="hfma")
    p.add_argument("--blocks", type=int, default=0)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--iters", type=int, default=1 << 22)
    p.add_argument("--unroll", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    if not args.no_build or not OUT_BIN.exists():
        build(args.nvcc, args.arch, args.use_fast_math)

    cmd = [
        str(OUT_BIN),
        "--op",
        args.op,
        "--blocks",
        str(args.blocks),
        "--threads",
        str(args.threads),
        "--iters",
        str(args.iters),
        "--unroll",
        str(args.unroll),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
    ]
    run(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()

