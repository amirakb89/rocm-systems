/******************************************************************************
 * MIT License
 * Copyright (c) 2025 DeepSeek
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include "../util.h"

#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define FINISHED_SUM_TAG 1024
static constexpr int32_t kWaveSize = 64;

using namespace rocshmem;
using gpu_bfloat16_t = hip_bfloat16;

/* ====================== Memory intrinsics (from DeepEP) ================== */

__device__ __forceinline__ int4 ld_nc_global(const int4* ptr) {
    int4 ret;
    ret.x = __builtin_nontemporal_load(&ptr->x);
    ret.y = __builtin_nontemporal_load(&ptr->y);
    ret.z = __builtin_nontemporal_load(&ptr->z);
    ret.w = __builtin_nontemporal_load(&ptr->w);
    return ret;
}
__device__ __forceinline__ int ld_nc_global(const int* ptr) {
    return __builtin_nontemporal_load(ptr);
}
__device__ __forceinline__ float ld_nc_global(const float* ptr) {
    return __builtin_nontemporal_load(ptr);
}

__device__ __forceinline__ void st_na_global(const int4* ptr, const int4& val) {
    auto* p = const_cast<int4*>(ptr);
    p->x = val.x; p->y = val.y; p->z = val.z; p->w = val.w;
}
__device__ __forceinline__ void st_na_global(const int* ptr, const int& val) {
    *const_cast<int*>(ptr) = val;
}

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC) \
{ \
    constexpr int _kStride = kWaveSize * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type _vals[(UNROLL_FACTOR)]; \
    auto _s = (SRC); auto _d = (DST); \
    for (int _i = (LANE_ID); _i < ((int)(N) / _kStride) * _kStride; _i += _kStride) { \
        _Pragma("unroll") for (int _j = 0; _j < (UNROLL_FACTOR); ++_j) \
            _vals[_j] = LD_FUNC(_s + _i + _j * kWaveSize); \
        _Pragma("unroll") for (int _j = 0; _j < (UNROLL_FACTOR); ++_j) \
            ST_FUNC(_d + _i + _j * kWaveSize, _vals[_j]); \
    } \
    for (int _i = ((int)(N) / _kStride) * _kStride + (LANE_ID); _i < (int)(N); _i += kWaveSize) \
        ST_FUNC(_d + _i, LD_FUNC(_s + _i)); \
}

#define SWITCH_HIDDEN(LAUNCH_CASE) \
    switch (hidden) { \
        case 2048: LAUNCH_CASE(2048); \
        case 4096: LAUNCH_CASE(4096); \
        case 5120: LAUNCH_CASE(5120); \
        case 7168: LAUNCH_CASE(7168); \
        default: ASSERT(false && "Unsupported hidden size"); \
    }

namespace ll_kernels {

template <typename T>
__host__ __device__ T cell_div(T a, T b) { return (a + b - 1) / b; }

__forceinline__ __device__ void warp_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

__forceinline__ __device__ void grid_barrier(int* global_counter, int num_blocks) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0)
    __hip_atomic_fetch_add(&global_counter[0], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  if (threadIdx.x == 0)
    while (__hip_atomic_load(global_counter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != num_blocks);
  __syncthreads();
}

__forceinline__ __device__ int warp_reduce_sum(int val) {
  for (int offset = kWaveSize / 2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}

__forceinline__ __device__ float quarter_warp_reduce_max(float val) {
  val = fmaxf(val, __shfl_xor(val, 8));
  val = fmaxf(val, __shfl_xor(val, 4));
  val = fmaxf(val, __shfl_xor(val, 2));
  val = fmaxf(val, __shfl_xor(val, 1));
  return val;
}

__forceinline__ __device__ float fp8_e4m3fnuz_to_float(uint8_t bits) {
#if defined(__gfx942__)
  return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(bits), 0);
#else
  if (bits == 0 || bits == 0x80) return 0.0f;
  int sign = (bits >> 7) & 1, exp = (bits >> 3) & 0xF, mant = bits & 0x7;
  float r = exp == 0 ? ldexpf((float)mant, -10) : ldexpf((float)(8 + mant), exp - 11);
  return sign ? -r : r;
#endif
}

/* ============================ DISPATCH ====================================
 *
 *  Phase 1:  BF16 → FP8 quantization + RDMA put
 *  Phase 2:  Receive raw FP8 via UNROLLED_WARP_COPY (no dequant)
 *            Stores FP8 bytes + scales into separate packed buffers.
 *            A standalone dequant kernel converts FP8→BF16 afterwards.
 *
 *========================================================================= */

template <int kNumWavesPerGroup, int kNumWaveGroups, int kHidden, typename T>
__global__ __launch_bounds__(kNumWavesPerGroup * kNumWaveGroups * kWaveSize, 1)
void dispatch_kernel(
    uint8_t *packed_recv_fp8, float *packed_recv_scales,
    int *packed_recv_src_info, int64_t *packed_recv_layout_range,
    int *packed_recv_count, int *global_atomic_counter,
    void *rdma_recv_x, int64_t *rdma_recv_count,
    void *rdma_x, const void *x, const int64_t *topk_idx,
    int *atomic_counter_per_expert, int *atomic_finish_counter_per_expert,
    int64_t *next_clean, int num_next_clean_int, int num_tokens,
    int num_topk, int num_experts, int rank, int num_ranks) {

  const int wg_id     = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int wave_id   = thread_id / kWaveSize;
  const int num_wgs   = static_cast<int>(gridDim.x);
  const int lane_id   = thread_id % kWaveSize;
  constexpr int num_waves   = kNumWaveGroups * kNumWavesPerGroup;
  constexpr int num_threads = kNumWaveGroups * kNumWavesPerGroup * kWaveSize;
  const int num_local_experts  = num_experts / num_ranks;
  const int wave_group_id      = wave_id / kNumWavesPerGroup;
  const int sub_wave_id        = wave_id % kNumWavesPerGroup;
  const int responsible_expert = wg_id * kNumWaveGroups + wave_group_id;

  constexpr int kNumPerChan = 128;
#if defined(__gfx942__)
  constexpr float kFP8Amax = 240.0f;
#else
  constexpr float kFP8Amax = 448.0f;
#endif
  constexpr float kFP8Margin = 1e-4f;
  constexpr int kElemsPerRead    = sizeof(int4) / sizeof(T);
  constexpr int kNumScales       = kHidden / kNumPerChan;
  constexpr size_t kFP8Bytes     = kHidden;
  constexpr size_t kMsgBytes     = sizeof(int4) + kFP8Bytes + kNumScales * sizeof(float);
  constexpr size_t kMsgInt4      = kMsgBytes / sizeof(int4);
  constexpr size_t kBF16Int4     = kHidden / kElemsPerRead;
  constexpr int kFP8Int4         = kHidden / 16;
  constexpr int kScalesInt4      = (kNumScales * sizeof(float)) / sizeof(int4);

  __shared__ int sh_expert_count[kNumWaveGroups];

  /* ---- Phase 1: FP8 cast + RDMA send ---- */
  if (wave_id < num_waves) {
    for (int token_idx = wg_id; token_idx < num_tokens; token_idx += num_wgs) {
      const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * kBF16Int4;
      auto hdr    = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_x) + token_idx * kMsgBytes);
      auto fp8out = reinterpret_cast<int2*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(int4));
      auto scl    = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(fp8out) + kFP8Bytes);

      auto dst_exp = wave_id < num_topk
          ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + wave_id)) : -1;
      if (thread_id == 0) hdr[0] = token_idx;

      #pragma unroll
      for (int i = thread_id; i < (int)kBF16Int4; i += num_threads) {
        auto v4 = __ldg(x_int4 + i);
        auto bf = reinterpret_cast<gpu_bfloat16_t*>(&v4);
        float fp[kElemsPerRead]; float amax = kFP8Margin;
        #pragma unroll
        for (int j = 0; j < kElemsPerRead; ++j) { fp[j] = static_cast<float>(bf[j]); amax = fmaxf(amax, fabsf(fp[j])); }
        amax = quarter_warp_reduce_max(amax);
        float scale = kFP8Amax / amax, scale_inv = amax / kFP8Amax;
        if (lane_id % 16 == 0) scl[i * kElemsPerRead / kNumPerChan] = scale_inv;

        int2 pk; auto fp8x2 = reinterpret_cast<__hip_fp8x2_storage_t*>(&pk);
        #pragma unroll
        for (int j = 0; j < kElemsPerRead; j += 2) {
          float2 sv = {fp[j] * scale, fp[j+1] * scale};
#if defined(__gfx942__)
          fp8x2[j/2] = __hip_cvt_float2_to_fp8x2(sv, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
#else
          fp8x2[j/2] = __hip_cvt_float2_to_fp8x2(sv, __HIP_SATFINITE, __HIP_E4M3);
#endif
        }
        fp8out[i] = pk;
      }
      __syncthreads();

      if (dst_exp >= 0) {
        int slot = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_exp, 1) : 0;
        slot = __shfl(slot, 0);
        const int dr = dst_exp / num_local_experts, dl = dst_exp % num_local_experts;
        const auto src = reinterpret_cast<uint64_t>(hdr);
        const auto dst = reinterpret_cast<uint64_t>(rdma_recv_x) +
            dl * num_ranks * num_tokens * kMsgBytes + rank * num_tokens * kMsgBytes + slot * kMsgBytes;

        if (dr != rank) {
          rocshmem_putmem_nbi_wave(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), kMsgBytes, dr);
        } else {
          UNROLLED_WARP_COPY(4, lane_id, kMsgInt4,
              reinterpret_cast<int4*>(dst), reinterpret_cast<const int4*>(src), ld_nc_global, st_na_global);
        }
        warp_sync();
        if (lane_id == 0)
          __hip_atomic_fetch_add(atomic_finish_counter_per_expert + dst_exp, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }

  /* ---- Last warp: count tokens per expert ---- */
  if (wave_id == num_waves - 1) {
    if (wg_id == 0) {
      for (int i = lane_id; i < num_next_clean_int; i += kWaveSize) next_clean[i] = 0;
      for (int i = lane_id; i < num_experts; i += kWaveSize)
        __hip_atomic_fetch_add(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    int ec[kNumWaveGroups] = {0};
    const int eb = wg_id * kNumWaveGroups, ee = min(eb + kNumWaveGroups, num_experts);
    for (int i = lane_id; i < num_tokens * num_topk; i += kWaveSize) {
      int idx = static_cast<int>(__ldg(topk_idx + i));
      if (idx >= eb && idx < ee) ec[idx - eb]++;
    }
    for (int i = eb; i < ee; ++i) {
      int s = warp_reduce_sum(ec[i - eb]);
      if (lane_id == 0) {
        sh_expert_count[i - eb] = s;
        __hip_atomic_fetch_add(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - s, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  __syncthreads();

  if (responsible_expert < num_experts && sub_wave_id == 0 && lane_id == 0) {
    const int dr = responsible_expert / num_local_experts, dl = responsible_expert % num_local_experts;
    const int ns = sh_expert_count[responsible_expert - wg_id * kNumWaveGroups];
    while (__hip_atomic_load(atomic_finish_counter_per_expert + responsible_expert, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != FINISHED_SUM_TAG * 2);
    if (dr != rank) rocshmem_long_atomic_add(rdma_recv_count + dl * num_ranks + rank, -ns - 1, dr);
    else __hip_atomic_store(rdma_recv_count + dl * num_ranks + rank, -ns - 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    atomic_counter_per_expert[responsible_expert] = 0;
    atomic_finish_counter_per_expert[responsible_expert] = 0;
    if (dr == 0) packed_recv_count[dl] = 0;
  }
  warp_sync();
  grid_barrier(global_atomic_counter, num_wgs);

  /* ---- Phase 2: Receive raw FP8 via UNROLLED_WARP_COPY (no dequant) ---- */
  if (responsible_expert < num_experts) {
    const int sr = responsible_expert / num_local_experts, le = responsible_expert % num_local_experts;
    const auto rb = reinterpret_cast<uint8_t*>(rdma_recv_x) +
        le * num_ranks * num_tokens * kMsgBytes + sr * num_tokens * kMsgBytes;

    const int stride_tokens = num_ranks * num_tokens;
    auto* psi = packed_recv_src_info + le * stride_tokens;
    auto* pr  = packed_recv_layout_range + le * num_ranks;

    auto* fp8_base  = packed_recv_fp8    + le * stride_tokens * kHidden;
    auto* scl_base  = packed_recv_scales + le * stride_tokens * kNumScales;

    __shared__ int sh_nr[kNumWaveGroups], sh_rb[kNumWaveGroups];
    int nrecv, rbegin;
    if (sub_wave_id == 0 && lane_id == 0) {
      while ((nrecv = __hip_atomic_load(reinterpret_cast<int*>(rdma_recv_count + le * num_ranks + sr), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) == 0);
      nrecv = -nrecv - 1;
      rbegin = atomicAdd(packed_recv_count + le, nrecv);
      sh_nr[wave_group_id] = nrecv; sh_rb[wave_group_id] = rbegin;
      pr[sr] = (static_cast<int64_t>(rbegin) << 32) | static_cast<int64_t>(nrecv);
    }
    __syncthreads();
    nrecv = sh_nr[wave_group_id]; rbegin = sh_rb[wave_group_id];

    for (int tok = sub_wave_id; tok < nrecv; tok += kNumWavesPerGroup) {
      const auto* h = reinterpret_cast<const int*>(rb + tok * kMsgBytes);
      if (lane_id == 0) psi[rbegin + tok] = ld_nc_global(h);

      const auto* fp8_src = reinterpret_cast<const int4*>(rb + tok * kMsgBytes + sizeof(int4));
      auto* fp8_dst = reinterpret_cast<int4*>(fp8_base + (rbegin + tok) * kHidden);
      UNROLLED_WARP_COPY(4, lane_id, kFP8Int4, fp8_dst, fp8_src, ld_nc_global, st_na_global);

      const auto* scl_src = reinterpret_cast<const int4*>(
          rb + tok * kMsgBytes + sizeof(int4) + kFP8Bytes);
      auto* scl_dst = reinterpret_cast<int4*>(scl_base + (rbegin + tok) * kNumScales);
      UNROLLED_WARP_COPY(1, lane_id, kScalesInt4, scl_dst, scl_src, ld_nc_global, st_na_global);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0)
    __hip_atomic_store(global_atomic_counter, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

/* Host-side dispatch launcher */
template <typename T>
void dispatch(uint8_t *packed_recv_fp8, float *packed_recv_scales,
    int* packed_recv_src_info, int64_t* packed_recv_layout_range,
    int* packed_recv_count, int* global_atomic_counter,
    void* rdma_recv_x, int64_t* rdma_recv_count,
    void* rdma_x, const void* x, const int64_t* topk_idx,
    int64_t* next_clean, int num_next_clean_int, int num_tokens, int hidden,
    int num_topk, int num_experts, int rank, int num_ranks,
    void* workspace, hipStream_t stream) {

  constexpr int kNumWavesPerGroup = 8, kNumWaveGroups = 2;
  ASSERT(num_topk <= 9 && hidden % 128 == 0);
  const auto num_wgs = cell_div(num_experts, kNumWaveGroups);
  const auto num_threads = kNumWaveGroups * kNumWavesPerGroup * kWaveSize;
  auto* ac = reinterpret_cast<int*>(workspace);
  auto* af = ac + num_experts;

#define DISPATCH_CASE(H) { \
  dispatch_kernel<kNumWavesPerGroup, kNumWaveGroups, H, T> \
    <<<num_wgs, num_threads, 0, stream>>>( \
      packed_recv_fp8, packed_recv_scales, \
      packed_recv_src_info, packed_recv_layout_range, \
      packed_recv_count, global_atomic_counter, rdma_recv_x, rdma_recv_count, \
      rdma_x, x, topk_idx, ac, af, next_clean, num_next_clean_int, \
      num_tokens, num_topk, num_experts, rank, num_ranks); } break

  SWITCH_HIDDEN(DISPATCH_CASE);
#undef DISPATCH_CASE
}

/* ======================== DEQUANT FP8 → BF16 ==============================
 *
 *  Standalone kernel: reads packed FP8 bytes + scales written by dispatch,
 *  dequantizes to BF16, writes to packed_recv_x for combine.
 *  Replaces the inline dequant that was previously in dispatch Phase 2.
 *
 *========================================================================= */

template <int kHidden, typename T>
__global__ void dequant_fp8_kernel(
    T* __restrict__ out,
    const uint8_t* __restrict__ fp8,
    const float* __restrict__ scales,
    const int* __restrict__ counts,
    int stride_tokens, int num_local_experts) {

  constexpr int kNumPerChan = 128;
  constexpr int kNumScales  = kHidden / kNumPerChan;
  constexpr int kFP8Int4    = kHidden / 16;

  const int expert = blockIdx.y;
  if (expert >= num_local_experts) return;
  const int tok = blockIdx.x;
  if (tok >= counts[expert]) return;

  const size_t base = (static_cast<size_t>(expert) * stride_tokens + tok) * kHidden;
  const size_t scl_base = (static_cast<size_t>(expert) * stride_tokens + tok) * kNumScales;

  const auto* f_src = reinterpret_cast<const int4*>(fp8 + base);
  auto* d_dst = reinterpret_cast<int4*>(out + base);
  const float* s_src = scales + scl_base;

  for (int e = threadIdx.x; e < kFP8Int4; e += blockDim.x) {
    int4 raw = ld_nc_global(f_src + e);
    const auto* bytes = reinterpret_cast<const uint8_t*>(&raw);
    float s = __ldg(s_src + (e * 16 / kNumPerChan));

    int4 lo, hi;
    auto* blo = reinterpret_cast<gpu_bfloat16_t*>(&lo);
    auto* bhi = reinterpret_cast<gpu_bfloat16_t*>(&hi);
    #pragma unroll
    for (int j = 0; j < 8; ++j) blo[j] = gpu_bfloat16_t(fp8_e4m3fnuz_to_float(bytes[j]) * s);
    #pragma unroll
    for (int j = 0; j < 8; ++j) bhi[j] = gpu_bfloat16_t(fp8_e4m3fnuz_to_float(bytes[8+j]) * s);
    st_na_global(d_dst + e * 2,     lo);
    st_na_global(d_dst + e * 2 + 1, hi);
  }
}

template <typename T>
void dequant_fp8_to_bf16(T* packed_recv_x,
    const uint8_t* packed_recv_fp8, const float* packed_recv_scales,
    const int* packed_recv_count,
    int num_local_experts, int stride_tokens, int hidden,
    hipStream_t stream) {

  constexpr int kBlockSize = 256;
  auto max_tokens = stride_tokens;

#define DEQUANT_CASE(H) { \
  dim3 grid(max_tokens, num_local_experts); \
  dequant_fp8_kernel<H, T><<<grid, kBlockSize, 0, stream>>>( \
      packed_recv_x, packed_recv_fp8, packed_recv_scales, \
      packed_recv_count, stride_tokens, num_local_experts); } break

  SWITCH_HIDDEN(DEQUANT_CASE);
#undef DEQUANT_CASE
}

/* ============================ COMBINE ===================================== */

template <int kNumWavesPerGroup, int kNumWaveGroups, int kNumMaxTopK, int kHidden, typename T>
__global__ __launch_bounds__(kNumWavesPerGroup * kNumWaveGroups * kWaveSize, 1)
void combine_kernel(T* combined_x, void* rdma_recv_x, int64_t* rdma_recv_flag,
    void* rdma_send_x, const void* x, const int64_t* topk_idx,
    const float* topk_weights, const int* src_info, const int64_t* layout_range,
    int* global_atomic_counter, int64_t* next_clean, int num_next_clean_int,
    int* atomic_clean_flag, int num_tokens, int num_topk,
    int num_experts, int rank, int num_ranks) {

  const int wg_id     = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int wave_id   = thread_id / kWaveSize;
  const int num_wgs   = static_cast<int>(gridDim.x);
  const int lane_id   = thread_id % kWaveSize;
  const int num_local_experts = num_experts / num_ranks;
  const int wave_group_id = wave_id / kNumWavesPerGroup;
  const int sub_wave_id   = wave_id % kNumWavesPerGroup;
  const int resp_exp = wg_id * kNumWaveGroups + wave_group_id;

  constexpr int kEPI4 = sizeof(int4) / sizeof(T);
  constexpr size_t kBI4 = kHidden / kEPI4;
  constexpr size_t kSlotHdr = sizeof(int4);
  constexpr size_t kSlotBytes = kSlotHdr + kHidden * sizeof(T);

  __syncthreads();
  constexpr int maxw = 16;
  __shared__ volatile int swc[maxw];
  if (thread_id < maxw) swc[thread_id] = 0;
  __syncthreads();

  /* ---- Phase 1: RDMA send BF16 ---- */
  if (wg_id == 0 && wave_group_id == 0 && sub_wave_id == 0) {
    for (int i = lane_id; i < num_next_clean_int; i += kWaveSize) next_clean[i] = 0;
    warp_sync();
    if (lane_id == 0) __hip_atomic_fetch_add(atomic_clean_flag, num_experts, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }

  if (resp_exp < num_experts) {
    const int dr = resp_exp / num_local_experts, le = resp_exp % num_local_experts;
    const int ge = rank * num_local_experts + le;
    const auto layout = __ldg(layout_range + le * num_ranks + dr);
    const auto* lx = reinterpret_cast<const int4*>(x) + le * num_ranks * num_tokens * kBI4;
    const auto* lsi = &src_info[le * num_ranks * num_tokens];
    auto* sb = reinterpret_cast<uint8_t*>(rdma_send_x) + le * num_ranks * num_tokens * kSlotBytes;

    int nts, off;
    auto* lp = reinterpret_cast<const int*>(&layout);
    nts = lp[0]; off = lp[1];

    for (int tok = off + sub_wave_id; tok < off + nts; tok += kNumWavesPerGroup) {
      const auto* xi4 = lx + tok * kBI4;
      auto* buf = reinterpret_cast<int4*>(sb + tok * kSlotBytes + kSlotHdr);
      int stk = __ldg(lsi + tok);
      auto dp = reinterpret_cast<uint64_t>(rdma_recv_x) + (ge * num_tokens + stk) * kSlotBytes + kSlotHdr;

      if (dr == rank) {
        UNROLLED_WARP_COPY(4, lane_id, kBI4, reinterpret_cast<int4*>(dp), xi4, ld_nc_global, st_na_global);
      } else {
        UNROLLED_WARP_COPY(4, lane_id, kBI4, buf, xi4, ld_nc_global, st_na_global);
        rocshmem_putmem_nbi_wave(reinterpret_cast<void*>(dp), reinterpret_cast<void*>(buf), kHidden * sizeof(T), dr);
      }
    }

    if (lane_id == 0) {
      __hip_atomic_fetch_add(const_cast<int*>(&swc[wave_group_id]), 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      warp_sync();
      while (swc[wave_group_id] < kNumWavesPerGroup);
    }

    if (sub_wave_id == 0 && lane_id == 0) {
      while (__hip_atomic_load(atomic_clean_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0);
      if (dr != rank) rocshmem_long_atomic_add(rdma_recv_flag + ge, 1, dr);
      else __hip_atomic_store(reinterpret_cast<int64_t*>(rdma_recv_flag + ge), (int64_t)1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_fetch_add(atomic_clean_flag, -1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  if (resp_exp < num_experts && sub_wave_id == 0 && lane_id == 0)
    while (__hip_atomic_load(reinterpret_cast<int64_t*>(rdma_recv_flag + resp_exp), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0);

  grid_barrier(global_atomic_counter, num_wgs);

  /* ---- Phase 2: Weighted reduction ---- */
  if (thread_id < (int)kBI4) {
    for (int ti = wg_id; ti < num_tokens; ti += num_wgs) {
      int ri[kNumMaxTopK]; float rw[kNumMaxTopK];
      #pragma unroll
      for (int k = 0; k < kNumMaxTopK; ++k) {
        if (k < num_topk) {
          ri[k] = static_cast<int>(__ldg(topk_idx + ti * num_topk + k));
          rw[k] = __ldg(topk_weights + ti * num_topk + k);
        } else { ri[k] = -1; rw[k] = 0.f; }
      }

      float c[kEPI4] = {0.f};
      #pragma unroll
      for (int k = 0; k < kNumMaxTopK; ++k) if (k < num_topk && ri[k] >= 0) {
        const auto sb = reinterpret_cast<const uint8_t*>(rdma_recv_x) +
            (ri[k] * num_tokens + ti) * kSlotBytes + kSlotHdr;
        int4 xv = ld_nc_global(reinterpret_cast<const int4*>(sb) + thread_id);
        auto* xb = reinterpret_cast<gpu_bfloat16_t*>(&xv);
        #pragma unroll
        for (int j = 0; j < kEPI4; ++j) c[j] += static_cast<float>(xb[j]) * rw[k];
      }

      int4 res; auto* rb = reinterpret_cast<gpu_bfloat16_t*>(&res);
      #pragma unroll
      for (int j = 0; j < kEPI4; ++j) rb[j] = gpu_bfloat16_t(c[j]);
      st_na_global(reinterpret_cast<int4*>(combined_x) + ti * kBI4 + thread_id, res);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0)
    __hip_atomic_store(global_atomic_counter, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

/* Host-side combine launcher */
template <typename T>
void combine(T* combined_x, void* rdma_recv_x, int64_t* rdma_recv_flag,
    void* rdma_send_x, const void* x, const int64_t* topk_idx,
    const float* topk_weights, const int* src_info, const int64_t* layout_range,
    int* global_atomic_counter, int64_t* next_clean, int num_next_clean_int,
    int num_tokens, int num_topk, int hidden, int num_experts, int rank,
    int num_ranks, void* workspace, hipStream_t stream) {

  constexpr int kNumWavesPerGroup = 8, kNumWaveGroups = 2, kMaxK = 9;
  ASSERT(num_topk <= kMaxK);
  const auto num_wgs = cell_div(num_experts, kNumWaveGroups);
  const auto num_threads = kNumWaveGroups * kNumWavesPerGroup * kWaveSize;
  auto* acf = reinterpret_cast<int*>(workspace);

#define COMBINE_CASE(H) { \
  combine_kernel<kNumWavesPerGroup, kNumWaveGroups, kMaxK, H, T> \
    <<<num_wgs, num_threads, 0, stream>>>( \
      combined_x, rdma_recv_x, rdma_recv_flag, rdma_send_x, x, topk_idx, \
      topk_weights, src_info, layout_range, global_atomic_counter, \
      next_clean, num_next_clean_int, acf, num_tokens, num_topk, \
      num_experts, rank, num_ranks); } break

  SWITCH_HIDDEN(COMBINE_CASE);
#undef COMBINE_CASE
}

}  // namespace ll_kernels
