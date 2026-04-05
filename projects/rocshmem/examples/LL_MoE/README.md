# LL_MoE: Low-Latency MoE All-to-All with rocSHMEM

This directory contains standalone implementations of **Mixture-of-Experts (MoE)
dispatch and combine kernels** using native rocSHMEM APIs. The kernels are ported
from the mori V1 low-latency (V1LL) internode dispatch/combine path, replacing
mori's internal SHMEM abstraction with direct `rocshmem_putmem_nbi_wave`,
`rocshmem_long_atomic_add`, and HIP atomics.

## Directory Structure

```
LL_MoE/
├── LL_MoE_Combine_V1LL.hpp      # Combine kernels + shared config/args structs
├── LL_MoE_Dispatch_V1LL.hpp      # Dispatch kernels (CopyToStaging, InterNode)
├── V1LL_combine_example.cpp      # C++ test harness for combine
├── V1LL_dispatch_example.cpp     # C++ test harness for dispatch
├── LL_MoE.hpp                    # Original MoE example header (refactored)
├── LL_MoE_Buffers.hpp            # Buffer management helpers
├── LL_MoE_Data.hpp               # Test data generation
├── LL_MoE_Kernels.hpp            # Original MoE kernels
├── MoE_example.cpp               # Original MoE example entry point
├── CMakeLists.txt                # CMake targets for all tests
├── python_wrapper/               # PyTorch extension + torchrun test
│   ├── csrc/
│   │   ├── v1ll_ops.cu           # Host-side kernel launchers + rocSHMEM wrappers
│   │   ├── v1ll_module.cu        # pybind11 module (V1LLBuffer class)
│   │   └── v1ll_api.hpp          # C++ API header
│   ├── setup.py                  # torch CUDAExtension build script
│   └── test_v1ll.py              # torchrun test: correctness, benchmark, Kineto
└── README.md                     # This file
```

## Prerequisites

- **ROCm** >= 6.0 (tested with 7.2)
- **rocSHMEM** built and installed (static library `librocshmem.a`)
- **MPI** (OpenMPI or MPICH) for the C++ tests
- **PyTorch** with ROCm support for the Python wrapper
- GPU architecture: gfx942 (MI300X) or adjust `TARGET_GPU_ARCH`

Environment variables used throughout:

| Variable | Default | Description |
|---|---|---|
| `ROCSHMEM_DIR` | `~/rocshmem` | Path to rocSHMEM install prefix |
| `ROCM_HOME` | `/opt/rocm` | Path to ROCm installation |
| `TARGET_GPU_ARCH` | `gfx942` | GPU architecture for hipcc `--offload-arch` |

---

## Option 1: C++ Tests (CMake + MPI)

### Build

```bash
# From the rocm-systems repo root, or any build directory
cd projects/rocshmem/examples

mkdir build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH="${ROCSHMEM_DIR:-$HOME/rocshmem};${ROCM_HOME:-/opt/rocm}" \
  -DCMAKE_CXX_COMPILER=${ROCM_HOME:-/opt/rocm}/bin/hipcc

make -j$(nproc) v1ll_dispatch_test v1ll_combine_test
```

This produces two executables: `LL_MoE/v1ll_dispatch_test` and
`LL_MoE/v1ll_combine_test`.

### Run

Both tests use MPI to launch one process per GPU:

```bash
# Dispatch test (8 GPUs)
mpirun -np 8 ./LL_MoE/v1ll_dispatch_test

# Combine test (8 GPUs)
mpirun -np 8 ./LL_MoE/v1ll_combine_test
```

Each test performs:
1. rocSHMEM initialization via MPI
2. Symmetric/device buffer allocation
3. Synthetic data generation (round-robin routing to all peers)
4. Kernel launch with correctness verification
5. Performance measurement using CUDA events

### Profile with rocprofv3

To avoid multi-rank SQLite write conflicts, profile only rank 0:

```bash
cat > /tmp/run_profiled.sh << 'SCRIPT'
#!/bin/bash
if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
    rocprofv3 --hip-activity on --kernel-trace on -o /tmp/v1ll_prof -- "$@"
else
    "$@"
fi
SCRIPT
chmod +x /tmp/run_profiled.sh

mpirun -np 8 /tmp/run_profiled.sh ./LL_MoE/v1ll_dispatch_test
```

---

## Option 2: Python Wrapper (PyTorch + torchrun)

The `python_wrapper/` directory provides a PyTorch C++ extension that exposes
the V1LL dispatch and combine kernels as a `V1LLBuffer` Python class, using
`torch.Tensor` for inputs/outputs and `torch.distributed` for bootstrapping
rocSHMEM.

### Build

```bash
cd projects/rocshmem/examples/LL_MoE/python_wrapper

# Set environment (adjust paths as needed)
export ROCSHMEM_DIR=${ROCSHMEM_DIR:-$HOME/rocshmem}
export TARGET_GPU_ARCH=gfx942

# Build and install in development mode
python setup.py develop
```

> **Note**: Use `python setup.py develop` (not `pip install -e .`) so that the
> system PyTorch with ROCm support is available during the build.

### Run

```bash
torchrun --nproc-per-node=8 test_v1ll.py
```

The test script runs three phases:

1. **Correctness tests** -- dispatch routing verification (per-token hidden
   data, expert indices, weights) and combine accumulation check against
   expected weighted sums.
2. **Mori-style benchmark** -- dispatch and combine are timed separately using
   CUDA events. Durations and bandwidths are all-gathered across ranks and
   reported per-round, following the same methodology as mori's
   `bench_dispatch_combine.py` (algorithmic BW with `ll_mode_scale`, bottleneck
   latency as max across ranks).
3. **Kineto profiling** -- per-kernel GPU timing via `torch.profiler` with
   optional chrome trace export for visualization in `chrome://tracing` or
   [Perfetto UI](https://ui.perfetto.dev/).

### CLI Options

```
torchrun --nproc-per-node=8 test_v1ll.py [OPTIONS]

Model parameters:
  --num-tokens N             Input tokens per rank (default: 128)
  --hidden N                 Hidden dimension (default: 7168)
  --num-topk K               Experts per token (default: 8)
  --num-experts-per-rank N   Experts per rank (default: 36)
  --max-tokens-per-rank N    Max tokens per rank for buffer sizing (default: 128)

Kernel launch config:
  --block-num N              Thread blocks (default: 32)
  --rdma-block-num N         RDMA blocks, 0 for single-node (default: 2)
  --warp-per-block N         Warps per block (default: 4)

Benchmark settings:
  --num-iters N              Benchmark iterations, all-gathered across ranks (default: 10)
  --warmup N                 Warmup iterations (default: 1)
  --graph-replay-iters N     CUDA event replays per benchmark iteration (default: 10)
  --num-kineto-tests N       Kineto profiling iterations (default: 30)
  --trace-dir PATH           Save chrome traces to this directory (default: none)
```

### Example with Traces

```bash
torchrun --nproc-per-node=8 test_v1ll.py \
  --num-tokens 128 \
  --hidden 7168 \
  --num-iters 10 \
  --graph-replay-iters 10 \
  --num-kineto-tests 30 \
  --trace-dir /tmp/v1ll_traces
```

Chrome traces are saved as `full_pipeline_rank{N}.json` and can be loaded in
`chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev/).

---

## Bandwidth Calculation

Bandwidth is computed identically to mori's `bench_dispatch_combine.py`:

```
total_bytes = total_recv_num_token * hidden_dim * element_size
bandwidth (GB/s) = total_bytes / 1e9 / (duration_ms / 1e3)
```

- `total_recv_num_token` is the **actual** received token count on each rank
- `element_size` is 2 bytes for bf16
- Duration is the average across `graph_replay_iters` CUDA event measurements
- **Algo BW** is the mean of per-rank BWs across all ranks
- **Latency** is the max duration across ranks (bottleneck)
- **`ll_mode_scale`** = `max_tokens_per_rank * num_experts_per_token / total_recv_tokens`,
  scales measured BW to estimate throughput at full buffer capacity

## Kernel Architecture

**Dispatch** (input tokens -> expert destinations):
1. `EpDispatchCopyToStaging` -- packs `[hidden | indices | weights | scales | src_tok_id]` into a flat staging buffer
2. `EpDispatchInterNodeV1KernelLowLatency` -- routes staged tokens to destination PEs:
   - *Intra-node blocks*: direct P2P copy via `rocshmem_ptr` + `memcpy`
   - *RDMA send blocks*: `rocshmem_putmem_nbi_wave` + `rocshmem_long_atomic_add` for signaling
   - *RDMA recv blocks*: spin-wait on atomic counters, then unpack
   - *Sync*: grid-wide barrier, exchange received token counts

**Combine** (expert outputs -> original token positions):
1. `EpCombineSync` + `EpCombineSyncBarrier` -- synchronize all PEs before combine
2. `EpCombineInterNodeV1KernelLowLatency` -- reverse routing with weighted accumulation
3. `EpCombineAll` -- final reduction of partial results

## Known Issues

- `rocshmem_finalize()` may SIGSEGV during process shutdown in multi-GPU PyTorch
  environments. The Python test wraps this in a `try-except` block; the crash
  occurs after all results are collected and is cosmetic.
