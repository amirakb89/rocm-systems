/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Host-side test for the mori V1LL combine kernel port (LL_MoE_Combine_V1LL.hpp).
 *
 * Allocates all symmetric + device buffers matching mori's layout, populates
 * them with synthetic dispatch-output data, runs the 4 combine kernels, and
 * verifies correctness.
 *
 * Build:  hipcc -I<rocshmem-include> V1LL_combine_example.cpp
 *         -lrocshmem -lmpi -o v1ll_combine_test
 *
 * Run:    mpirun -np 8 ./v1ll_combine_test
 *
 *****************************************************************************/

#include <mpi.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <rocshmem/rocshmem.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "LL_MoE_Combine_V1LL.hpp"

using namespace rocshmem;
using namespace mori_v1ll;

#define CHECK_HIP(call)                                            \
  do {                                                             \
    hipError_t err = (call);                                       \
    if (err != hipSuccess) {                                       \
      fprintf(stderr, "HIP error %d at %s:%d\n", err, __FILE__,   \
              __LINE__);                                           \
      exit(err);                                                   \
    }                                                              \
  } while (0)

/* ========================================================================== */
/*  Helper: allocate symmetric memory, zero it, return as SymmPtr             */
/* ========================================================================== */
static SymmPtr shmem_alloc(size_t bytes, hipStream_t stream) {
  SymmPtr p;
  if (bytes == 0) return p;
  p.localPtr = rocshmem_malloc(bytes);
  if (!p.localPtr) {
    fprintf(stderr, "rocshmem_malloc(%zu) failed\n", bytes);
    exit(EXIT_FAILURE);
  }
  CHECK_HIP(hipMemsetAsync(p.localPtr, 0, bytes, stream));
  return p;
}

/* ========================================================================== */
/*  Helper: hipMalloc + zero, returns typed pointer                           */
/* ========================================================================== */
template <typename U>
static U* device_alloc(size_t count, hipStream_t stream) {
  U* ptr = nullptr;
  if (count == 0) return ptr;
  CHECK_HIP(hipMalloc(&ptr, count * sizeof(U)));
  CHECK_HIP(hipMemsetAsync(ptr, 0, count * sizeof(U), stream));
  return ptr;
}

/* ========================================================================== */
/*  CombineV1LLTest — sets up all buffers and runs the combine               */
/* ========================================================================== */
template <typename T>
class CombineV1LLTest {
 public:
  CombineV1LLTest(int numTokens, int hiddenDim, int numExpertPerToken,
                   int numExpertPerRank, int maxNumInpTokenPerRank,
                   int warpPerBlock, int blockNum, int rdmaBlockNum)
      : numTokens_(numTokens),
        hiddenDim_(hiddenDim),
        numExpertPerToken_(numExpertPerToken),
        numExpertPerRank_(numExpertPerRank),
        maxNumInpTokenPerRank_(maxNumInpTokenPerRank),
        warpPerBlock_(warpPerBlock),
        blockNum_(blockNum),
        rdmaBlockNum_(rdmaBlockNum) {
    comm_init();
    CHECK_HIP(hipStreamCreate(&stream_));

    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, deviceId_));
    multiProcessorCount_ = prop.multiProcessorCount;

    build_config();
    allocate_buffers();
    setup_test_data();
  }

  ~CombineV1LLTest() {
    free_buffers();
    CHECK_HIP(hipStreamDestroy(stream_));
    comm_finalize();
  }

  void run() {
    LaunchCombineV1LL<T>(args_, blockNum_, warpPerBlock_, rdmaBlockNum_,
                         multiProcessorCount_, stream_);
    CHECK_HIP(hipStreamSynchronize(stream_));
  }

  bool verify() {
    const size_t total = (size_t)numTokens_ * hiddenDim_;
    std::vector<T> h_out(total);
    T* combineOutLocal =
        reinterpret_cast<T*>(args_.interNodeV1TokBufs.combineOut.localPtr);
    CHECK_HIP(hipMemcpy(h_out.data(), combineOutLocal, total * sizeof(T),
                        hipMemcpyDeviceToHost));

    std::vector<T> h_inp(total);
    CHECK_HIP(hipMemcpy(h_inp.data(), args_.inpTokenBuf, total * sizeof(T),
                        hipMemcpyDeviceToHost));

    int mismatches = 0;
    for (int t = 0; t < numTokens_; ++t) {
      for (int h = 0; h < hiddenDim_; ++h) {
        float expected = static_cast<float>(h_inp[t * hiddenDim_ + h]);
        float actual   = static_cast<float>(h_out[t * hiddenDim_ + h]);
        float err = fabsf(actual - expected);
        float rel = (fabsf(expected) > 1e-6f) ? err / fabsf(expected) : err;
        if (rel > 0.01f && err > 1e-4f) {
          if (mismatches < 5)
            printf("[rank %d] MISMATCH token=%d h=%d expected=%.4f actual=%.4f\n",
                   rank_, t, h, expected, actual);
          ++mismatches;
        }
      }
    }

    if (mismatches == 0) {
      printf("[rank %d] PASSED (%d tokens, %d hidden)\n",
             rank_, numTokens_, hiddenDim_);
    } else {
      printf("[rank %d] FAILED — %d / %zu mismatches\n",
             rank_, mismatches, total);
    }
    return mismatches == 0;
  }

  int rank() const { return rank_; }

 private:
  /* ---- MPI + rocSHMEM init/finalize (same pattern as LL_MoE.hpp) ---- */
  void comm_init() {
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

    int deviceCount;
    CHECK_HIP(hipGetDeviceCount(&deviceCount));
    deviceId_ = rank_ % deviceCount;
    CHECK_HIP(hipSetDevice(deviceId_));

    rocshmem_uniqueid_t uid;
    rocshmem_init_attr_t attr;
    if (rank_ == 0) rocshmem_get_uniqueid(&uid);
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    rocshmem_set_attr_uniqueid_args(rank_, worldSize_, &uid, &attr);
    rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);

    gpuPerNode_ = worldSize_;  // single-node assumption
    nNodes_     = 1;
  }

  void comm_finalize() {
    rocshmem_finalize();
    MPI_Finalize();
  }

  /* ---- Build EpDispatchCombineConfig ---- */
  void build_config() {
    auto& c            = cfg_;
    c.rank             = rank_;
    c.worldSize        = worldSize_;
    c.hiddenDim        = hiddenDim_;
    c.scaleDim         = 0;
    c.scaleTypeSize    = 0;
    c.maxTokenTypeSize = (int)sizeof(T);
    c.maxNumInpTokenPerRank = maxNumInpTokenPerRank_;
    c.numExpertPerRank = numExpertPerRank_;
    c.numExpertPerToken = numExpertPerToken_;
    c.maxTotalRecvTokens = 0;
    c.warpNumPerBlock  = warpPerBlock_;
    c.blockNum         = blockNum_;
    c.useExternalInpBuffer = true;
    c.gpuPerNode       = gpuPerNode_;
    c.rdmaBlockNum     = rdmaBlockNum_;
    c.numQpPerPe       = 1;
  }

  /* ---- Allocate symmetric + device buffers (mirrors mori's Handle ctor) ---- */
  void allocate_buffers() {
    const auto& c = cfg_;
    const int nNodes = nNodes_;

    /* combXferBytes: hidden data + optional weights (mirrors DEF_COMMON_VARS) */
    const size_t combXferBytesPerTok =
        c.HiddenBytes(sizeof(T)) + c.WeightBytes();

    const size_t combineOutSize =
        (size_t)c.MaxNumTokensToSendPerRank() * c.hiddenDim * c.maxTokenTypeSize;
    const size_t dispatchOutSize =
        (size_t)c.MaxNumTokensToRecv() * c.hiddenDim * c.maxTokenTypeSize;
    const size_t maxStagingXfer =
        (size_t)c.MaxNumTokensToRecv() * combXferBytesPerTok;

    /* ---- Symmetric token buffers (ShmemBufsInterNodeV1) ---- */
    const size_t dispInpSize =
        (size_t)nNodes * c.MaxNumTokensToSendPerRank() * combXferBytesPerTok;
    const size_t stagingSize =
        (size_t)2 * nNodes * c.MaxNumTokensToSendPerRank() * combXferBytesPerTok;

    args_.interNodeV1TokBufs.dispatchInp = shmem_alloc(dispInpSize, stream_);
    args_.interNodeV1TokBufs.combineInp  = shmem_alloc(maxStagingXfer, stream_);
    args_.interNodeV1TokBufs.staging     = shmem_alloc(stagingSize, stream_);
    args_.interNodeV1TokBufs.dispatchOut = shmem_alloc(dispatchOutSize, stream_);
    args_.interNodeV1TokBufs.combineOut  = shmem_alloc(combineOutSize, stream_);

    /* ---- Symmetric weights / indices / scales ---- */
    const size_t maxWeightSize =
        (size_t)c.MaxNumTokensToRecv() * c.numExpertPerToken * sizeof(float);
    args_.shmemInpWeightsMemObj         = shmem_alloc(maxWeightSize, stream_);
    args_.shmemDispatchOutWeightsMemObj = shmem_alloc(maxWeightSize, stream_);
    args_.shmemCombineOutWeightsMemObj  = shmem_alloc(maxWeightSize, stream_);

    const size_t maxIndicesSize =
        (size_t)c.MaxNumTokensToRecv() * c.numExpertPerToken * sizeof(index_t);
    args_.shmemInpIndicesMemObj  = shmem_alloc(maxIndicesSize, stream_);
    args_.shmemOutIndicesMemObj  = shmem_alloc(maxIndicesSize, stream_);

    /* ---- Symmetric signaling ---- */
    const size_t sigSize = (size_t)c.worldSize * sizeof(index_t) * 2 * c.numQpPerPe;
    args_.recvTokenNumMemObj    = shmem_alloc(sigSize, stream_);
    args_.sendTokenNumMemObj    = shmem_alloc(sigSize, stream_);
    args_.sendAtomicSignalMemObj =
        shmem_alloc((size_t)(c.worldSize * 2) * sizeof(int64_t) * 2, stream_);

    const size_t nodeTokenNumSigSize = (size_t)nNodes * sizeof(uint64_t);
    args_.nodeRecvTokenNumMemObj = shmem_alloc(nodeTokenNumSigSize, stream_);

    args_.dispTokOffsetMemObj    = shmem_alloc(sizeof(index_t), stream_);

    const size_t maxNumOutToken =
        (size_t)c.MaxNumTokensToSend() * c.numExpertPerRank;
    args_.dispTokIdToSrcTokIdMemObj =
        shmem_alloc(maxNumOutToken * sizeof(index_t), stream_);

    const size_t barrierSize = (size_t)c.worldSize * sizeof(uint32_t);
    args_.crossDeviceBarrierMemObj =
        shmem_alloc(barrierSize * 2 * sizeof(uint64_t), stream_);

    const size_t chunkFlagSize =
        (size_t)nNodes * c.MaxNumTokensToSendPerRank() * sizeof(uint64_t);
    args_.interNodeChunkFlagMemObj = shmem_alloc(chunkFlagSize, stream_);

    /* ---- Device-local buffers ---- */
    args_.inpTokenBuf =
        device_alloc<T>((size_t)c.MaxNumTokensToRecv() * c.hiddenDim, stream_);
    args_.weightsBuf =
        device_alloc<float>((size_t)c.MaxNumTokensToRecv() * c.numExpertPerToken, stream_);
    args_.tokenIndices =
        device_alloc<index_t>((size_t)c.MaxNumTokensToSendPerRank() * c.numExpertPerToken, stream_);

    args_.dispatchGridBarrier = device_alloc<uint32_t>(c.worldSize, stream_);
    args_.combineGridBarrier  = device_alloc<uint32_t>(c.worldSize, stream_);

    args_.destPeTokenCounter  = device_alloc<index_t>(c.worldSize, stream_);
    args_.localPeTokenCounter = device_alloc<index_t>(c.worldSize, stream_);

    args_.dispReceiverIdxMap  = device_alloc<index_t>(maxNumOutToken, stream_);
    args_.dispSenderIdxMap    = device_alloc<index_t>(maxNumOutToken, stream_);
    args_.destPeTokenIdxMap   = device_alloc<index_t>(maxNumOutToken, stream_);
    args_.srcPeTokenIdxMap    = device_alloc<index_t>(maxNumOutToken, stream_);

    args_.dispDestTokIdMap = device_alloc<index_t>(maxNumOutToken, stream_);

    args_.totalRecvTokenNum = device_alloc<index_t>(1, stream_);

    args_.crossDeviceBarrierFlag = device_alloc<uint64_t>(1, stream_);

    args_.blockFlagCounter = device_alloc<index_t>(nNodes, stream_);
    args_.interNodeBlocksBarrier = device_alloc<uint32_t>(4, stream_);

    const size_t interNodeTokMap =
        (size_t)nNodes * c.MaxNumTokensToSendPerRank() * c.numExpertPerToken;
    args_.interNodeDispDestTokIdMap = device_alloc<index_t>(interNodeTokMap, stream_);

    args_.interNodeChunkFlagCombine =
        device_alloc<index_t>((size_t)nNodes * c.MaxNumTokensToSendPerRank(), stream_);

    args_.interNodeDispSendMap =
        device_alloc<index_t>((size_t)nNodes * c.MaxNumTokensToSendPerRank(), stream_);

    args_.destNodeTokenCounter = device_alloc<index_t>(nNodes, stream_);

    /* ---- Scalars ---- */
    args_.config       = cfg_;
    args_.rdmaBlockNum = rdmaBlockNum_;
    args_.curRankNumToken = numTokens_;

    CHECK_HIP(hipStreamSynchronize(stream_));
  }

  /* ---- Free everything ---- */
  void free_buffers() {
    auto sfree = [](SymmPtr& p) {
      if (p.localPtr) { rocshmem_free(p.localPtr); p.localPtr = nullptr; }
    };
    auto dfree = [](auto*& p) {
      if (p) { (void)hipFree(p); p = nullptr; }
    };

    sfree(args_.interNodeV1TokBufs.dispatchInp);
    sfree(args_.interNodeV1TokBufs.combineInp);
    sfree(args_.interNodeV1TokBufs.staging);
    sfree(args_.interNodeV1TokBufs.dispatchOut);
    sfree(args_.interNodeV1TokBufs.combineOut);
    sfree(args_.shmemInpWeightsMemObj);
    sfree(args_.shmemDispatchOutWeightsMemObj);
    sfree(args_.shmemCombineOutWeightsMemObj);
    sfree(args_.shmemInpIndicesMemObj);
    sfree(args_.shmemOutIndicesMemObj);
    sfree(args_.recvTokenNumMemObj);
    sfree(args_.sendTokenNumMemObj);
    sfree(args_.sendAtomicSignalMemObj);
    sfree(args_.nodeRecvTokenNumMemObj);
    sfree(args_.dispTokOffsetMemObj);
    sfree(args_.dispTokIdToSrcTokIdMemObj);
    sfree(args_.crossDeviceBarrierMemObj);
    sfree(args_.interNodeChunkFlagMemObj);

    dfree(args_.inpTokenBuf);
    dfree(args_.weightsBuf);
    dfree(args_.tokenIndices);
    dfree(args_.dispatchGridBarrier);
    dfree(args_.combineGridBarrier);
    dfree(args_.destPeTokenCounter);
    dfree(args_.localPeTokenCounter);
    dfree(args_.dispReceiverIdxMap);
    dfree(args_.dispSenderIdxMap);
    dfree(args_.destPeTokenIdxMap);
    dfree(args_.srcPeTokenIdxMap);
    dfree(args_.dispDestTokIdMap);
    dfree(args_.totalRecvTokenNum);
    dfree(args_.crossDeviceBarrierFlag);
    dfree(args_.blockFlagCounter);
    dfree(args_.interNodeBlocksBarrier);
    dfree(args_.interNodeDispDestTokIdMap);
    dfree(args_.interNodeChunkFlagCombine);
    dfree(args_.interNodeDispSendMap);
    dfree(args_.destNodeTokenCounter);
  }

  /* ==================================================================== */
  /*  Setup synthetic test data                                           */
  /*                                                                      */
  /*  Simulates a completed dispatch where every token on this rank       */
  /*  routed all experts to itself (all-local).  With deduplication,      */
  /*  only the first expert per token gets a valid slot.                  */
  /*                                                                      */
  /*  After combine, combined[t][h] == inpTokenBuf[t][h]                  */
  /*  (exactly one non-null expert contribution with weight 1.0).         */
  /* ==================================================================== */
  void setup_test_data() {
    const auto& c = cfg_;

    /* -- inpTokenBuf: fill with known pattern -- */
    {
      const size_t total = (size_t)numTokens_ * hiddenDim_;
      std::vector<T> h_inp(total);
      for (int t = 0; t < numTokens_; ++t)
        for (int h = 0; h < hiddenDim_; ++h)
          h_inp[t * hiddenDim_ + h] =
              static_cast<T>(1.0f + 0.001f * t + 0.0001f * (h % 64));
      CHECK_HIP(hipMemcpy(args_.inpTokenBuf, h_inp.data(),
                           total * sizeof(T), hipMemcpyHostToDevice));
    }

    /* -- weightsBuf: all 1.0 -- */
    {
      const size_t total = (size_t)numTokens_ * c.numExpertPerToken;
      std::vector<float> h_wt(total, 1.0f);
      CHECK_HIP(hipMemcpy(args_.weightsBuf, h_wt.data(),
                           total * sizeof(float), hipMemcpyHostToDevice));
    }

    /* -- tokenIndices: all experts on local rank -- */
    {
      const size_t total = (size_t)numTokens_ * c.numExpertPerToken;
      std::vector<index_t> h_idx(total);
      for (int t = 0; t < numTokens_; ++t)
        for (int e = 0; e < c.numExpertPerToken; ++e)
          h_idx[t * c.numExpertPerToken + e] =
              rank_ * c.numExpertPerRank + (e % c.numExpertPerRank);
      CHECK_HIP(hipMemcpy(args_.tokenIndices, h_idx.data(),
                           total * sizeof(index_t), hipMemcpyHostToDevice));
    }

    /* -- dispDestTokIdMap: first expert → valid slot, rest → null (dedup) -- */
    {
      const size_t maxOutTok = (size_t)c.MaxNumTokensToSend() * c.numExpertPerRank;
      std::vector<index_t> h_map(maxOutTok, NullFlatTokenIndex_());
      for (int t = 0; t < numTokens_; ++t) {
        h_map[t * c.numExpertPerToken + 0] = FlatTokenIndex_(rank_, t);
      }
      CHECK_HIP(hipMemcpy(args_.dispDestTokIdMap, h_map.data(),
                           maxOutTok * sizeof(index_t), hipMemcpyHostToDevice));
    }

    /* -- totalRecvTokenNum: this rank received numTokens tokens -- */
    {
      index_t val = numTokens_;
      CHECK_HIP(hipMemcpy(args_.totalRecvTokenNum, &val, sizeof(index_t),
                           hipMemcpyHostToDevice));
    }

    /* -- crossDeviceBarrierFlag: initial value 1 (V1/V1LL convention) -- */
    {
      uint64_t val = 1;
      CHECK_HIP(hipMemcpy(args_.crossDeviceBarrierFlag, &val, sizeof(uint64_t),
                           hipMemcpyHostToDevice));
    }

    /* -- crossDeviceBarrierMemObj: pre-fill with 1 so barrier unblocks -- */
    {
      const size_t n = (size_t)c.worldSize * 2;
      std::vector<uint64_t> h_bar(n, 1ULL);
      CHECK_HIP(hipMemcpy(args_.crossDeviceBarrierMemObj.localPtr, h_bar.data(),
                           n * sizeof(uint64_t), hipMemcpyHostToDevice));
    }

    CHECK_HIP(hipStreamSynchronize(stream_));
    rocshmem_barrier_all();
  }

  /* ---- Helpers for host-side index computation ---- */
  index_t FlatTokenIndex_(int pe, int localTokId) const {
    return pe * cfg_.MaxNumTokensToSend() + localTokId;
  }
  index_t NullFlatTokenIndex_() const {
    return cfg_.worldSize * cfg_.MaxNumTokensToSend();
  }

  /* ---- Member data ---- */
  int rank_{0}, worldSize_{0}, deviceId_{0};
  int gpuPerNode_{0}, nNodes_{0};
  int multiProcessorCount_{0};

  int numTokens_, hiddenDim_, numExpertPerToken_, numExpertPerRank_;
  int maxNumInpTokenPerRank_, warpPerBlock_, blockNum_, rdmaBlockNum_;

  EpDispatchCombineConfig cfg_{};
  EpDispatchCombineArgs<T> args_{};
  hipStream_t stream_{nullptr};
};

/* ========================================================================== */
/*  main                                                                      */
/* ========================================================================== */
int main(int argc, char** argv) {
  int numTokens  = 64;
  int hiddenDim  = 4096;
  int numExpertPerToken = 2;
  int numExpertPerRank  = 1;
  int maxTokPerRank     = 128;
  int warpPerBlock      = 4;
  int blockNum          = 32;
  int rdmaBlockNum      = 2;
  int numIterations     = 20;

  int opt;
  while ((opt = getopt(argc, argv, "n:h:k:e:b:r:w:i:")) != -1) {
    switch (opt) {
      case 'n': numTokens         = atoi(optarg); break;
      case 'h': hiddenDim         = atoi(optarg); break;
      case 'k': numExpertPerToken = atoi(optarg); break;
      case 'e': numExpertPerRank  = atoi(optarg); break;
      case 'b': blockNum          = atoi(optarg); break;
      case 'r': rdmaBlockNum      = atoi(optarg); break;
      case 'w': warpPerBlock      = atoi(optarg); break;
      case 'i': numIterations     = atoi(optarg); break;
      default:
        fprintf(stderr,
                "Usage: %s [-n tokens] [-h hidden] [-k topk] "
                "[-e experts/rank] [-b blocks] [-r rdma_blocks] "
                "[-w warps/block] [-i iterations]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
  }

  using T = hip_bfloat16;

  CombineV1LLTest<T> test(numTokens, hiddenDim, numExpertPerToken,
                           numExpertPerRank, maxTokPerRank, warpPerBlock,
                           blockNum, rdmaBlockNum);

  printf("[rank %d] Running V1LL combine (tokens=%d hidden=%d topk=%d "
         "blocks=%d rdma=%d wpb=%d iters=%d) ...\n",
         test.rank(), numTokens, hiddenDim, numExpertPerToken,
         blockNum, rdmaBlockNum, warpPerBlock, numIterations);

  /* Correctness check */
  test.run();
  bool ok = test.verify();
  if (!ok) return EXIT_FAILURE;

  /* Warmup */
  constexpr int warmupIters = 5;
  for (int i = 0; i < warmupIters; ++i) test.run();

  /* Benchmark */
  int benchIters = std::max(1, numIterations - warmupIters);
  std::vector<float> latency_us(benchIters);

  for (int i = 0; i < benchIters; ++i) {
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    test.run();
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float ms = 0.f;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    latency_us[i] = ms * 1000.f;

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
  }

  auto avg = [](const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.f) / (float)v.size();
  };
  float avg_us = avg(latency_us);
  float min_us = *std::min_element(latency_us.begin(), latency_us.end());
  float max_us = *std::max_element(latency_us.begin(), latency_us.end());

  size_t combBytesPerToken = (size_t)hiddenDim * sizeof(T);
  size_t totalSelections   = (size_t)numTokens * numExpertPerToken;
  double totalBytes        = (double)(totalSelections * combBytesPerToken);
  double bw_gbps           = totalBytes / 1e9 / (avg_us / 1e6);

  printf("[rank %d] Combine V1LL: avg=%.1f us, min=%.1f us, max=%.1f us "
         "(%d iters), BW=%.2f GB/s (%zu B/token)\n",
         test.rank(), avg_us, min_us, max_us, benchIters,
         bw_gbps, combBytesPerToken);
  fflush(stdout);

  return EXIT_SUCCESS;
}
