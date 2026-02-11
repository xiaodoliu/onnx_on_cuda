#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>

static void die(const char* msg) {
  std::fprintf(stderr, "ERROR: %s\n", msg);
  std::exit(1);
}

static void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

enum class Op : int {
  // fp16 / half2 arithmetic (hardware)
  hadd,
  hmul,
  hfma,
  h2add,
  h2mul,
  h2fma,

  // float libdevice / math
  expf_std,
  expf_fast,
  logf_std,
  sinf_std,
  sinf_fast,
  cosf_std,
  cosf_fast,
};

static Op parse_op(const std::string& s) {
  if (s == "hadd") return Op::hadd;
  if (s == "hmul") return Op::hmul;
  if (s == "hfma") return Op::hfma;
  if (s == "h2add") return Op::h2add;
  if (s == "h2mul") return Op::h2mul;
  if (s == "h2fma") return Op::h2fma;

  if (s == "expf") return Op::expf_std;
  if (s == "__expf") return Op::expf_fast;
  if (s == "logf") return Op::logf_std;
  if (s == "sinf") return Op::sinf_std;
  if (s == "__sinf") return Op::sinf_fast;
  if (s == "cosf") return Op::cosf_std;
  if (s == "__cosf") return Op::cosf_fast;

  die("unknown --op. Try: hadd hmul hfma h2add h2mul h2fma expf __expf logf sinf __sinf cosf __cosf");
  return Op::hadd;
}

__device__ __forceinline__ __half op_half(Op op, __half x, __half c) {
  switch (op) {
    case Op::hadd: return __hadd(x, c);
    case Op::hmul: return __hmul(x, c);
    case Op::hfma: return __hfma(x, c, c);
    default: return x;
  }
}

__device__ __forceinline__ __half2 op_half2(Op op, __half2 x, __half2 c) {
  switch (op) {
    case Op::h2add: return __hadd2(x, c);
    case Op::h2mul: return __hmul2(x, c);
    case Op::h2fma: return __hfma2(x, c, c);
    default: return x;
  }
}

__device__ __forceinline__ float op_float(Op op, float x) {
  switch (op) {
    case Op::expf_std: return expf(x);
    case Op::expf_fast: return __expf(x);
    case Op::logf_std: return logf(x);
    case Op::sinf_std: return sinf(x);
    case Op::sinf_fast: return __sinf(x);
    case Op::cosf_std: return cosf(x);
    case Op::cosf_fast: return __cosf(x);
    default: return x;
  }
}

template <int UNROLL>
__global__ void bench_kernel(Op op, std::uint64_t iters, float* out_f, __half* out_h, __half2* out_h2) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Make inputs depend on thread id so compiler can't constant-fold.
  float xf = 0.001f * (1.0f + (tid % 1024));
  __half xh = __float2half_rn(xf);
  __half2 xh2 = __floats2half2_rn(xf, xf + 0.0007f);

  // Constants for fp16 ops.
  __half ch = __float2half_rn(0.1234f);
  __half2 ch2 = __floats2half2_rn(0.1234f, 0.4321f);

  // Unrolled inner loop: each step depends on previous value.
  for (std::uint64_t i = 0; i < iters; i += UNROLL) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      // Guard for iters not divisible by UNROLL.
      if (i + static_cast<std::uint64_t>(u) >= iters) break;

      // Select the right data type for the op.
      if (op == Op::hadd || op == Op::hmul || op == Op::hfma) {
        xh = op_half(op, xh, ch);
      } else if (op == Op::h2add || op == Op::h2mul || op == Op::h2fma) {
        xh2 = op_half2(op, xh2, ch2);
      } else {
        xf = op_float(op, xf);
      }
    }
  }

  // Store one value per thread (prevents dead-code elimination).
  if (out_f) out_f[tid] = xf;
  if (out_h) out_h[tid] = xh;
  if (out_h2) out_h2[tid] = xh2;
}

static double ops_per_iter(Op op) {
  switch (op) {
    case Op::h2add:
    case Op::h2mul:
    case Op::h2fma:
      return 2.0;  // half2 does 2 lanes
    default:
      return 1.0;
  }
}

int main(int argc, char** argv) {
  std::string op_s = "hfma";
  int blocks = 0;          // 0 => auto
  int threads = 256;
  std::uint64_t iters = 1ull << 22;  // per thread
  int unroll = 4;
  int warmup = 1;
  int runs = 5;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--op") && i + 1 < argc) op_s = argv[++i];
    else if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--unroll") && i + 1 < argc) unroll = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
      std::printf(
          "Usage: %s --op <name> [--blocks N] [--threads N] [--iters N] [--unroll N] [--warmup N] [--runs N]\n"
          "Ops:\n"
          "  fp16:  hadd hmul hfma  h2add h2mul h2fma\n"
          "  math:  expf __expf logf  sinf __sinf  cosf __cosf\n",
          argv[0]);
      return 0;
    } else {
      die("unknown argument (use --help)");
    }
  }

  if (threads <= 0 || threads > 1024) die("--threads must be 1..1024");
  if (unroll != 1 && unroll != 2 && unroll != 4 && unroll != 8) die("--unroll must be 1,2,4,8");
  if (runs <= 0) die("--runs must be > 0");
  if (warmup < 0) die("--warmup must be >= 0");

  Op op = parse_op(op_s);

  int dev = 0;
  cuda_check(cudaSetDevice(dev), "cudaSetDevice");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");

  if (blocks == 0) {
    // A reasonable default: enough blocks to fill the GPU, but not too large.
    blocks = prop.multiProcessorCount * 20;
  }

  const int64_t nthreads = static_cast<int64_t>(blocks) * static_cast<int64_t>(threads);
  if (nthreads <= 0) die("invalid launch configuration");

  float* d_out_f = nullptr;
  __half* d_out_h = nullptr;
  __half2* d_out_h2 = nullptr;

  const bool is_half = (op == Op::hadd || op == Op::hmul || op == Op::hfma);
  const bool is_half2 = (op == Op::h2add || op == Op::h2mul || op == Op::h2fma);

  if (is_half) cuda_check(cudaMalloc(&d_out_h, sizeof(__half) * nthreads), "cudaMalloc(out_h)");
  else if (is_half2) cuda_check(cudaMalloc(&d_out_h2, sizeof(__half2) * nthreads), "cudaMalloc(out_h2)");
  else cuda_check(cudaMalloc(&d_out_f, sizeof(float) * nthreads), "cudaMalloc(out_f)");

  auto launch = [&]() {
    switch (unroll) {
      case 1: bench_kernel<1><<<blocks, threads>>>(op, iters, d_out_f, d_out_h, d_out_h2); break;
      case 2: bench_kernel<2><<<blocks, threads>>>(op, iters, d_out_f, d_out_h, d_out_h2); break;
      case 4: bench_kernel<4><<<blocks, threads>>>(op, iters, d_out_f, d_out_h, d_out_h2); break;
      case 8: bench_kernel<8><<<blocks, threads>>>(op, iters, d_out_f, d_out_h, d_out_h2); break;
      default: die("unreachable unroll");
    }
  };

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  // Warmup (no timing).
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  cuda_check(cudaGetLastError(), "kernel launch (warmup)");
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize (warmup)");

  // Timed runs.
  double best_ms = 1e100;
  double sum_ms = 0.0;
  for (int r = 0; r < runs; ++r) {
    cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    launch();
    cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    cuda_check(cudaGetLastError(), "kernel launch (timed)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    best_ms = (ms < best_ms) ? ms : best_ms;
    sum_ms += ms;
  }

  // Copy back one element so the runtime definitely "uses" results.
  if (d_out_f) {
    float h = 0;
    cuda_check(cudaMemcpy(&h, d_out_f, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(out_f[0])");
  } else if (d_out_h) {
    __half h{};
    cuda_check(cudaMemcpy(&h, d_out_h, sizeof(__half), cudaMemcpyDeviceToHost), "cudaMemcpy(out_h[0])");
  } else if (d_out_h2) {
    __half2 h{};
    cuda_check(cudaMemcpy(&h, d_out_h2, sizeof(__half2), cudaMemcpyDeviceToHost), "cudaMemcpy(out_h2[0])");
  }

  const double total_ops = static_cast<double>(nthreads) * static_cast<double>(iters) * ops_per_iter(op);
  const double best_s = best_ms / 1e3;
  const double avg_s = (sum_ms / runs) / 1e3;

  std::printf("GPU: %s (SMs=%d)\n", prop.name, prop.multiProcessorCount);
  std::printf("op=%s blocks=%d threads=%d iters=%llu unroll=%d warmup=%d runs=%d\n",
              op_s.c_str(), blocks, threads,
              static_cast<unsigned long long>(iters), unroll, warmup, runs);
  std::printf("best_ms=%.3f avg_ms=%.3f\n", best_ms, (sum_ms / runs));
  std::printf("throughput(best)=%.3f GOP/s  throughput(avg)=%.3f GOP/s\n",
              (total_ops / best_s) / 1e9, (total_ops / avg_s) / 1e9);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_out_f);
  cudaFree(d_out_h);
  cudaFree(d_out_h2);
  return 0;
}

