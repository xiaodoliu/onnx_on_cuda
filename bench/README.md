## CUDA libdevice / fp16 microbench

This folder contains a tiny CUDA microbenchmark to measure raw throughput of:

- fp16 / half2 arithmetic intrinsics (`hadd`, `hmul`, `hfma`, `h2add`, `h2mul`, `h2fma`)
- common libdevice math for `float` (`expf`, `__expf`, `logf`, `sinf`, `__sinf`, `cosf`, `__cosf`)

It also contains an ONNX+ORT “mathbench” generator so you can measure the same kinds of ops
**through an ONNX model** (more realistic execution path for ORT CUDA EP).

### Build + run (recommended)

RTX 4090 is `sm_89`:

```bash
python3 bench/run_bench.py --op hfma
python3 bench/run_bench.py --op h2fma
python3 bench/run_bench.py --op __expf --use_fast_math
python3 bench/run_bench.py --op expf
```

### ONNX fp16 mathbench (run via ORT CUDA)

First install `onnx` in your venv:

```bash
python -m pip install onnx
```

Generate fp16 ONNX models:

```bash
python3 bench/make_fp16_mathbench_onnx.py --op add --shape 1,4096,4096 --depth 256 --out bench/models/fp16_add.onnx
python3 bench/make_fp16_mathbench_onnx.py --op mul --shape 1,4096,4096 --depth 256 --out bench/models/fp16_mul.onnx
python3 bench/make_fp16_mathbench_onnx.py --op fma --shape 1,4096,4096 --depth 256 --out bench/models/fp16_fma.onnx
python3 bench/make_fp16_mathbench_onnx.py --op exp --shape 1,4096,4096 --depth 64  --out bench/models/fp16_exp.onnx
python3 bench/make_fp16_mathbench_onnx.py --op sin --shape 1,4096,4096 --depth 64  --out bench/models/fp16_sin.onnx

# More “realistic transformer blocks” (elementwise + reduction, heavy libdevice usage)
python3 bench/make_fp16_mathbench_onnx.py --op softmax  --shape 1,4096,4096 --depth 64 --axis -1 --out bench/models/fp16_softmax.onnx
python3 bench/make_fp16_mathbench_onnx.py --op layernorm --shape 1,4096,4096 --depth 64 --axis -1 --epsilon 1e-5 --out bench/models/fp16_layernorm.onnx
python3 bench/make_fp16_mathbench_onnx.py --op rmsnorm   --shape 1,4096,4096 --depth 64 --axis -1 --epsilon 1e-5 --out bench/models/fp16_rmsnorm.onnx
python3 bench/make_fp16_mathbench_onnx.py --op gelu      --shape 1,4096,4096 --depth 64 --out bench/models/fp16_gelu.onnx
python3 bench/make_fp16_mathbench_onnx.py --op silu      --shape 1,4096,4096 --depth 64 --out bench/models/fp16_silu.onnx
```

Run (end-to-end ORT time, includes any H2D/D2H copies):

```bash
python3 bench/run_fp16_mathbench_ort.py --model bench/models/fp16_add.onnx --count_ops 1
python3 bench/run_fp16_mathbench_ort.py --model bench/models/fp16_fma.onnx --count_ops 2
python3 bench/run_fp16_mathbench_ort.py --model bench/models/fp16_exp.onnx --count_ops 1

# realistic blocks: count_ops is not “true FLOPs” here; prefer comparing latency + kernel mix
python3 bench/run_fp16_mathbench_ort.py --model bench/models/fp16_softmax.onnx --count_ops 1
python3 bench/run_fp16_mathbench_ort.py --model bench/models/fp16_layernorm.onnx --count_ops 1
python3 bench/run_fp16_mathbench_ort.py --model bench/models/fp16_gelu.onnx --count_ops 1
```

Profile with Nsight Systems to see the GPU kernels:

```bash
nsys profile --trace=cuda,osrt --force-overwrite=true -o fp16_exp_ort \
  python3 bench/run_fp16_mathbench_ort.py --model bench/models/fp16_exp.onnx --count_ops 1 --warmup 2 --iters 3
```

### Sample results

These are example outputs on **NVIDIA GeForce RTX 4090 Laptop GPU** using ORT CUDA EP.

Softmax (large):

```bash
python bench/run_fp16_mathbench_ort.py --model bench/models/fp16_softmax.onnx --warmup 5 --iters 20
```

```text
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
model=bench/models/fp16_softmax.onnx
input=X shape=[1, 4096, 4096] dtype=fp16 elems=16777216
output=Y_softmax dtype=tensor(float16)
warmup=5 iters=20 disable_optimizations=False
depth=64 count_ops(per_element_per_layer)=1.0
best_s=0.069219 avg_s=0.071167
effective_throughput(best)=15.512 GOP/s  effective_throughput(avg)=15.088 GOP/s
```

Softmax (small sanity check):

```bash
python bench/run_fp16_mathbench_ort.py --model bench/models/fp16_softmax_small.onnx --warmup 3 --iters 10
```

```text
Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
model=bench/models/fp16_softmax_small.onnx
input=X shape=[1, 256, 512] dtype=fp16 elems=131072
output=Y_softmax dtype=tensor(float16)
warmup=3 iters=10 disable_optimizations=False
depth=2 count_ops(per_element_per_layer)=1.0
best_s=0.000153 avg_s=0.000162
effective_throughput(best)=1.714 GOP/s  effective_throughput(avg)=1.617 GOP/s
```

### Run the binary directly

```bash
./bench/bench_libdevice --help
./bench/bench_libdevice --op h2fma --iters 4194304 --threads 256 --blocks 0 --runs 5
```

### Notes

- `--blocks 0` auto-selects `SMs * 20`, which is usually enough to saturate the GPU.
- The benchmark reports **best** and **average** throughput in GOP/s over multiple runs.
- For `half2` ops, the tool counts **2 ops/iter** (2 lanes).

