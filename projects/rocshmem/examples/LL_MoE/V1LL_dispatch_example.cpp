/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Host-side test for the mori V1LL dispatch kernel port.
 *
 * Each rank has numTokens tokens with numExpertPerToken expert assignments.
 * Experts are round-robin distributed across PEs so that each token sends
 * to a different intra-node peer.  After dispatch, each PE verifies that
 * it received the expected tokens (hidden data, indices, weights) on its
 * dispatchOut buffer.
 *
 * Build:  via CMake (add_executable in LL_MoE/CMakeLists.txt)
 * Run:    mpirun -np 8 ./v1ll_dispatch_test
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

#include "LL_MoE_Dispatch_V1LL.hpp"

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

template <typename U>
static U* device_alloc(size_t count, hipStream_t stream) {
  U* ptr = nullptr;
  if (count == 0) return ptr;
  CHECK_HIP(hipMalloc(&ptr, count * sizeof(U)));
  CHECK_HIP(hipMemsetAsync(ptr, 0, count * sizeof(U), stream));
  return ptr;
}

template <typename T>
class DispatchV1LLTest {
 public:
  DispatchV1LLTest(int numTokens, int hiddenDim, int numExpertPerToken,
                    int numExpertPerRank, int maxNumInpTokenPerRank,
                    int warpPerBlock, int blockNum, int rdmaBlockNum)
      : numTokens_(numTokens), hiddenDim_(hiddenDim),
        numExpertPerToken_(numExpertPerToken),
        numExpertPerRank_(numExpertPerRank),
        maxNumInpTokenPerRank_(maxNumInpTokenPerRank),
        warpPerBlock_(warpPerBlock), blockNum_(blockNum),
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

  ~DispatchV1LLTest() {
    free_buffers();
    CHECK_HIP(hipStreamDestroy(stream_));
    comm_finalize();
  }

  void run() {
    LaunchDispatchV1LL<T>(args_, blockNum_, warpPerBlock_, rdmaBlockNum_,
                          multiProcessorCount_, stream_);
    CHECK_HIP(hipStreamSynchronize(stream_));
  }

  bool verify() {
    const auto& c = cfg_;
    index_t h_totalRecv = 0;
    CHECK_HIP(hipMemcpy(&h_totalRecv, args_.totalRecvTokenNum,
                         sizeof(index_t), hipMemcpyDeviceToHost));

    const int expectedRecv = numTokens_ * worldSize_;
    if (h_totalRecv != expectedRecv) {
      printf("[rank %d] FAILED — received %d tokens, expected %d\n",
             rank_, h_totalRecv, expectedRecv);
      return false;
    }

    const size_t maxRecv = (size_t)c.MaxNumTokensToRecv();

    std::vector<T> h_dispOut(maxRecv * hiddenDim_, T{0});
    CHECK_HIP(hipMemcpy(h_dispOut.data(),
                         args_.interNodeV1TokBufs.dispatchOut.localPtr,
                         maxRecv * hiddenDim_ * sizeof(T),
                         hipMemcpyDeviceToHost));

    std::vector<index_t> h_outIndices(maxRecv * numExpertPerToken_, 0);
    CHECK_HIP(hipMemcpy(h_outIndices.data(),
                         args_.shmemOutIndicesMemObj.localPtr,
                         maxRecv * numExpertPerToken_ * sizeof(index_t),
                         hipMemcpyDeviceToHost));

    std::vector<float> h_outWeights(maxRecv * numExpertPerToken_, 0.f);
    CHECK_HIP(hipMemcpy(h_outWeights.data(),
                         args_.shmemDispatchOutWeightsMemObj.localPtr,
                         maxRecv * numExpertPerToken_ * sizeof(float),
                         hipMemcpyDeviceToHost));

    std::vector<index_t> h_srcTokIdMap(maxRecv, 0);
    CHECK_HIP(hipMemcpy(h_srcTokIdMap.data(),
                         args_.dispTokIdToSrcTokIdMemObj.localPtr,
                         maxRecv * sizeof(index_t),
                         hipMemcpyDeviceToHost));

    int hiddenMismatches = 0, indexMismatches = 0, weightMismatches = 0;
    int tokensChecked = 0;

    for (int t = 0; t < h_totalRecv; ++t) {
      index_t flatSrcTokId = h_srcTokIdMap[t];
      int srcPe    = flatSrcTokId / c.MaxNumTokensToSend();
      int srcTokId = flatSrcTokId % c.MaxNumTokensToSend();

      if (srcPe < 0 || srcPe >= worldSize_ ||
          srcTokId < 0 || srcTokId >= numTokens_) {
        if (hiddenMismatches < 3)
          printf("[rank %d] token %d: invalid srcPe=%d srcTokId=%d (flat=%d)\n",
                 rank_, t, srcPe, srcTokId, flatSrcTokId);
        ++hiddenMismatches;
        continue;
      }

      for (int h = 0; h < hiddenDim_; ++h) {
        float expected = 1.0f + 0.001f * srcTokId + 0.0001f * (h % 64);
        float actual   = static_cast<float>(h_dispOut[t * hiddenDim_ + h]);
        if (fabsf(actual - expected) > 0.02f) {
          if (hiddenMismatches < 5)
            printf("[rank %d] token %d h %d: got %.4f expected %.4f "
                   "(src rank %d tok %d)\n",
                   rank_, t, h, actual, expected, srcPe, srcTokId);
          ++hiddenMismatches;
        }
      }

      for (int e = 0; e < c.numExpertPerToken; ++e) {
        int expectedIdx = ((srcPe + e) % worldSize_) * c.numExpertPerRank;
        int actualIdx   = h_outIndices[t * c.numExpertPerToken + e];
        if (actualIdx != expectedIdx) {
          if (indexMismatches < 5)
            printf("[rank %d] token %d exp %d: index got %d expected %d\n",
                   rank_, t, e, actualIdx, expectedIdx);
          ++indexMismatches;
        }
      }

      for (int e = 0; e < c.numExpertPerToken; ++e) {
        float w = h_outWeights[t * c.numExpertPerToken + e];
        if (fabsf(w - 1.0f) > 1e-5f) {
          if (weightMismatches < 5)
            printf("[rank %d] token %d exp %d: weight got %.4f expected 1.0\n",
                   rank_, t, e, w);
          ++weightMismatches;
        }
      }
      ++tokensChecked;
    }

    int totalErrors = hiddenMismatches + indexMismatches + weightMismatches;
    if (totalErrors == 0) {
      printf("[rank %d] PASSED — verified %d/%d recv tokens "
             "(hidden, indices, weights all correct)\n",
             rank_, tokensChecked, h_totalRecv);
    } else {
      printf("[rank %d] FAILED — %d hidden errors, %d index errors, "
             "%d weight errors in %d tokens\n",
             rank_, hiddenMismatches, indexMismatches,
             weightMismatches, h_totalRecv);
    }
    return totalErrors == 0;
  }

  void reset_buffers() {
    CHECK_HIP(hipMemsetAsync(args_.totalRecvTokenNum, 0, sizeof(index_t), stream_));
    CHECK_HIP(hipMemsetAsync(args_.destPeTokenCounter, 0,
                              cfg_.worldSize * sizeof(index_t), stream_));
    CHECK_HIP(hipMemsetAsync(args_.dispatchGridBarrier, 0,
                              cfg_.worldSize * sizeof(uint32_t), stream_));
    CHECK_HIP(hipMemsetAsync(args_.combineGridBarrier, 0,
                              cfg_.worldSize * sizeof(uint32_t), stream_));
    CHECK_HIP(hipMemsetAsync(args_.blockFlagCounter, 0,
                              nNodes_ * sizeof(index_t), stream_));
    CHECK_HIP(hipMemsetAsync(args_.interNodeBlocksBarrier, 0,
                              4 * sizeof(uint32_t), stream_));
    size_t maxOutTok = (size_t)cfg_.MaxNumTokensToSend() * cfg_.numExpertPerRank;
    CHECK_HIP(hipMemsetAsync(args_.dispDestTokIdMap, 0,
                              maxOutTok * sizeof(index_t), stream_));
    CHECK_HIP(hipMemsetAsync(args_.dispTokOffsetMemObj.localPtr, 0,
                              sizeof(index_t), stream_));
    CHECK_HIP(hipStreamSynchronize(stream_));
    rocshmem_barrier_all();
  }

  int rank() const { return rank_; }

 private:
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

    gpuPerNode_ = worldSize_;
    nNodes_     = 1;
  }

  void comm_finalize() {
    rocshmem_finalize();
    MPI_Finalize();
  }

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

  void allocate_buffers() {
    const auto& c = cfg_;
    const int nNodes = nNodes_;
    const size_t xferBytesPerTok = c.XferBytesPerToken(sizeof(T));
    const size_t combXferBytesPerTok = c.HiddenBytes(sizeof(T)) + c.WeightBytes();

    const size_t dispatchOutSize =
        (size_t)c.MaxNumTokensToRecv() * c.hiddenDim * c.maxTokenTypeSize;
    const size_t combineOutSize =
        (size_t)c.MaxNumTokensToSendPerRank() * c.hiddenDim * c.maxTokenTypeSize;
    const size_t stagingSize =
        (size_t)c.MaxNumTokensToSendPerRank() * xferBytesPerTok;
    const size_t dispInpSize =
        (size_t)nNodes * c.MaxNumTokensToSendPerRank() * xferBytesPerTok;
    const size_t combStagingSize =
        (size_t)2 * nNodes * c.MaxNumTokensToSendPerRank() * combXferBytesPerTok;

    args_.interNodeV1TokBufs.dispatchInp = shmem_alloc(dispInpSize, stream_);
    args_.interNodeV1TokBufs.combineInp  = shmem_alloc(
        (size_t)c.MaxNumTokensToRecv() * combXferBytesPerTok, stream_);
    args_.interNodeV1TokBufs.staging     = shmem_alloc(
        std::max(stagingSize, combStagingSize), stream_);
    args_.interNodeV1TokBufs.dispatchOut = shmem_alloc(dispatchOutSize, stream_);
    args_.interNodeV1TokBufs.combineOut  = shmem_alloc(combineOutSize, stream_);

    const size_t maxWeightSize =
        (size_t)c.MaxNumTokensToRecv() * c.numExpertPerToken * sizeof(float);
    args_.shmemInpWeightsMemObj         = shmem_alloc(maxWeightSize, stream_);
    args_.shmemDispatchOutWeightsMemObj = shmem_alloc(maxWeightSize, stream_);
    args_.shmemCombineOutWeightsMemObj  = shmem_alloc(maxWeightSize, stream_);

    const size_t maxIndicesSize =
        (size_t)c.MaxNumTokensToRecv() * c.numExpertPerToken * sizeof(index_t);
    args_.shmemInpIndicesMemObj = shmem_alloc(maxIndicesSize, stream_);
    args_.shmemOutIndicesMemObj = shmem_alloc(maxIndicesSize, stream_);

    const size_t sigSize = (size_t)c.worldSize * sizeof(index_t) * 2 * c.numQpPerPe;
    args_.recvTokenNumMemObj    = shmem_alloc(sigSize, stream_);
    args_.sendTokenNumMemObj    = shmem_alloc(sigSize, stream_);
    args_.sendAtomicSignalMemObj =
        shmem_alloc((size_t)(c.worldSize * 2) * sizeof(int64_t) * 2, stream_);

    args_.nodeRecvTokenNumMemObj = shmem_alloc((size_t)nNodes * sizeof(uint64_t), stream_);
    args_.dispTokOffsetMemObj    = shmem_alloc(sizeof(index_t), stream_);

    const size_t maxNumOutToken =
        (size_t)c.MaxNumTokensToSend() * c.numExpertPerRank;
    args_.dispTokIdToSrcTokIdMemObj =
        shmem_alloc(maxNumOutToken * sizeof(index_t), stream_);

    args_.crossDeviceBarrierMemObj =
        shmem_alloc((size_t)c.worldSize * 2 * sizeof(uint64_t), stream_);
    args_.interNodeChunkFlagMemObj =
        shmem_alloc((size_t)nNodes * c.MaxNumTokensToSendPerRank() * sizeof(uint64_t), stream_);

    args_.inpTokenBuf =
        device_alloc<T>((size_t)c.MaxNumTokensToSendPerRank() * c.hiddenDim, stream_);
    args_.weightsBuf =
        device_alloc<float>((size_t)c.MaxNumTokensToSendPerRank() * c.numExpertPerToken, stream_);
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
    args_.dispDestTokIdMap    = device_alloc<index_t>(maxNumOutToken, stream_);
    args_.totalRecvTokenNum   = device_alloc<index_t>(1, stream_);
    args_.crossDeviceBarrierFlag = device_alloc<uint64_t>(1, stream_);
    args_.blockFlagCounter    = device_alloc<index_t>(nNodes, stream_);
    args_.interNodeBlocksBarrier = device_alloc<uint32_t>(4, stream_);

    const size_t interNodeTokMap =
        (size_t)nNodes * c.MaxNumTokensToSendPerRank() * c.numExpertPerToken;
    args_.interNodeDispDestTokIdMap = device_alloc<index_t>(interNodeTokMap, stream_);
    args_.interNodeChunkFlagCombine =
        device_alloc<index_t>((size_t)nNodes * c.MaxNumTokensToSendPerRank(), stream_);
    args_.interNodeDispSendMap =
        device_alloc<index_t>((size_t)nNodes * c.MaxNumTokensToSendPerRank(), stream_);
    args_.destNodeTokenCounter = device_alloc<index_t>(nNodes, stream_);

    args_.config       = cfg_;
    args_.rdmaBlockNum = rdmaBlockNum_;
    args_.curRankNumToken = numTokens_;

    uint64_t barrierVal = 1;
    CHECK_HIP(hipMemcpy(args_.crossDeviceBarrierFlag, &barrierVal,
                         sizeof(uint64_t), hipMemcpyHostToDevice));

    CHECK_HIP(hipStreamSynchronize(stream_));
  }

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

  void setup_test_data() {
    const auto& c = cfg_;

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

    {
      const size_t total = (size_t)numTokens_ * c.numExpertPerToken;
      std::vector<float> h_wt(total, 1.0f);
      CHECK_HIP(hipMemcpy(args_.weightsBuf, h_wt.data(),
                           total * sizeof(float), hipMemcpyHostToDevice));
    }

    {
      const size_t total = (size_t)numTokens_ * c.numExpertPerToken;
      std::vector<index_t> h_idx(total);
      for (int t = 0; t < numTokens_; ++t)
        for (int e = 0; e < c.numExpertPerToken; ++e) {
          int destPe = (rank_ + e) % worldSize_;
          h_idx[t * c.numExpertPerToken + e] = destPe * c.numExpertPerRank;
        }
      CHECK_HIP(hipMemcpy(args_.tokenIndices, h_idx.data(),
                           total * sizeof(index_t), hipMemcpyHostToDevice));
    }

    CHECK_HIP(hipStreamSynchronize(stream_));
    rocshmem_barrier_all();
  }

  int rank_{0}, worldSize_{0}, deviceId_{0};
  int gpuPerNode_{0}, nNodes_{0};
  int multiProcessorCount_{0};
  int numTokens_, hiddenDim_, numExpertPerToken_, numExpertPerRank_;
  int maxNumInpTokenPerRank_, warpPerBlock_, blockNum_, rdmaBlockNum_;
  EpDispatchCombineConfig cfg_{};
  EpDispatchCombineArgs<T> args_{};
  hipStream_t stream_{nullptr};
};

int main(int argc, char** argv) {
  int numTokens  = 128;
  int hiddenDim  = 7168;
  int numExpertPerToken = 8;
  int numExpertPerRank  = 36;
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

  using DType = hip_bfloat16;

  DispatchV1LLTest<DType> test(numTokens, hiddenDim, numExpertPerToken,
                                numExpertPerRank, maxTokPerRank, warpPerBlock,
                                blockNum, rdmaBlockNum);

  printf("[rank %d] Running V1LL dispatch (tokens=%d hidden=%d topk=%d "
         "blocks=%d rdma=%d wpb=%d iters=%d) ...\n",
         test.rank(), numTokens, hiddenDim, numExpertPerToken,
         blockNum, rdmaBlockNum, warpPerBlock, numIterations);

  test.run();
  bool ok = test.verify();
  if (!ok) return EXIT_FAILURE;

  constexpr int warmupIters = 5;
  for (int i = 0; i < warmupIters; ++i) {
    test.reset_buffers();
    test.run();
  }

  int benchIters = std::max(1, numIterations - warmupIters);
  std::vector<float> latency_us(benchIters);

  for (int i = 0; i < benchIters; ++i) {
    test.reset_buffers();

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

  float avg_us = std::accumulate(latency_us.begin(), latency_us.end(), 0.f) /
                 (float)latency_us.size();
  float min_us = *std::min_element(latency_us.begin(), latency_us.end());
  float max_us = *std::max_element(latency_us.begin(), latency_us.end());

  size_t dispBytesPerTok = (size_t)hiddenDim * sizeof(DType);
  size_t totalSelections = (size_t)numTokens * numExpertPerToken;
  double totalBytes      = (double)(totalSelections * dispBytesPerTok);
  double bw_gbps         = totalBytes / 1e9 / (avg_us / 1e6);

  printf("[rank %d] Dispatch V1LL: avg=%.1f us, min=%.1f us, max=%.1f us "
         "(%d iters), BW=%.2f GB/s (%zu B/token)\n",
         test.rank(), avg_us, min_us, max_us, benchIters,
         bw_gbps, dispBytesPerTok);
  fflush(stdout);

  return EXIT_SUCCESS;
}
