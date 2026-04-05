/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Port of mori's InterNodeV1LL combine kernels to direct rocSHMEM APIs.
 * Maintains the exact same data layout, struct definitions, and algorithmic
 * logic as mori's EpCombineInterNodeV1KernelLowLatency path.
 *
 * SymmMemObjPtr  -> SymmPtr (local ptr + rocshmem_ptr for P2P resolution)
 * mori::shmem::* -> rocshmem_* device APIs
 * mori::core::*  -> __hip_atomic_* intrinsics
 *
 *****************************************************************************/

#pragma once

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cassert>

using namespace rocshmem;

namespace mori_v1ll {

static constexpr int32_t kWaveSize = 64;
using index_t = int32_t;

/* ========================================================================== */
/*  SymmPtr: drop-in for mori's SymmMemObjPtr using rocSHMEM symmetric heap   */
/* ========================================================================== */
struct SymmPtr {
  void* localPtr{nullptr};

  __host__ __device__ bool IsValid() const { return localPtr != nullptr; }

  template <typename T>
  __device__ T GetAs() const { return reinterpret_cast<T>(localPtr); }

  template <typename T>
  __device__ T GetAs(int pe) const {
    return reinterpret_cast<T>(rocshmem_ptr(localPtr, pe));
  }
};

/* ========================================================================== */
/*  Config: mirrors mori::moe::EpDispatchCombineConfig                        */
/* ========================================================================== */
struct EpDispatchCombineConfig {
  int rank{0};
  int worldSize{0};
  int hiddenDim{4096};
  int scaleDim{32};
  int scaleTypeSize{1};
  int maxTokenTypeSize{4};
  int maxNumInpTokenPerRank{128};
  int numExpertPerRank{1};
  int numExpertPerToken{2};
  int maxTotalRecvTokens{0};
  int warpNumPerBlock{1};
  int blockNum{1};
  bool useExternalInpBuffer{true};
  int gpuPerNode{8};
  int rdmaBlockNum{1};
  int numQpPerPe{1};

  __host__ __device__ int MaxNumTokensToSendPerRank() const {
    return maxNumInpTokenPerRank;
  }
  __host__ __device__ int MaxNumTokensToSend() const {
    return worldSize * MaxNumTokensToSendPerRank();
  }
  __host__ __device__ int MaxNumTokensToRecvPerRank() const {
    if (maxTotalRecvTokens > 0) {
      int perRank = (maxTotalRecvTokens + worldSize - 1) / worldSize;
      return perRank < maxNumInpTokenPerRank ? perRank : maxNumInpTokenPerRank;
    }
    return maxNumInpTokenPerRank;
  }
  __host__ __device__ int MaxNumTokensToRecv() const {
    return worldSize * MaxNumTokensToRecvPerRank();
  }
  __host__ __device__ size_t HiddenBytes(size_t typeSize) const {
    return typeSize * hiddenDim;
  }
  __host__ __device__ size_t IndexBytes() const {
    return numExpertPerToken * sizeof(index_t);
  }
  __host__ __device__ size_t WeightBytes() const {
    return numExpertPerToken * sizeof(float);
  }
  __host__ __device__ size_t SrcTokenIdBytes() const { return sizeof(index_t); }
  __host__ __device__ size_t ScaleBytes() const { return scaleDim * scaleTypeSize; }
  __host__ __device__ size_t XferBytesPerToken(size_t typeSize) const {
    return HiddenBytes(typeSize) + IndexBytes() + WeightBytes() + SrcTokenIdBytes() + ScaleBytes();
  }
};

/* ========================================================================== */
/*  Buffer group: mirrors mori::moe::ShmemBufsInterNodeV1                     */
/* ========================================================================== */
struct ShmemBufsInterNodeV1 {
  SymmPtr dispatchInp;
  SymmPtr combineInp;
  SymmPtr dispatchOut;
  SymmPtr combineOut;
  SymmPtr staging;
};

/* ========================================================================== */
/*  Args struct: mirrors mori::moe::EpDispatchCombineArgs<T>                  */
/*  SymmMemObjPtr fields replaced with SymmPtr.                               */
/* ========================================================================== */
template <typename T>
struct EpDispatchCombineArgs {
  using data_type = T;

  EpDispatchCombineConfig config;
  int rdmaBlockNum{-1};
  index_t curRankNumToken{0};
  index_t* tokenIndices{nullptr};
  T* inpTokenBuf{nullptr};
  T* outTokenBuf{nullptr};
  float* weightsBuf{nullptr};
  uint8_t* scalesBuf{nullptr};

  ShmemBufsInterNodeV1 interNodeV1TokBufs;

  SymmPtr shmemInpWeightsMemObj;
  SymmPtr shmemDispatchOutWeightsMemObj;
  SymmPtr shmemCombineOutWeightsMemObj;
  SymmPtr shmemInpScalesMemObj;
  SymmPtr shmemOutScalesMemObj;
  SymmPtr shmemInpIndicesMemObj;
  SymmPtr shmemOutIndicesMemObj;
  SymmPtr recvTokenNumMemObj;
  SymmPtr sendTokenNumMemObj;
  SymmPtr sendAtomicSignalMemObj;

  uint32_t* dispatchGridBarrier{nullptr};
  uint32_t* combineGridBarrier{nullptr};
  index_t* destPeTokenCounter{nullptr};
  index_t* localPeTokenCounter{nullptr};
  index_t* dispReceiverIdxMap{nullptr};
  index_t* dispSenderIdxMap{nullptr};
  index_t* destPeTokenIdxMap{nullptr};
  index_t* srcPeTokenIdxMap{nullptr};

  SymmPtr dispTokOffsetMemObj;
  SymmPtr dispTokIdToSrcTokIdMemObj;

  index_t* dispDestTokIdMap{nullptr};
  index_t* totalRecvTokenNum{nullptr};

  SymmPtr crossDeviceBarrierMemObj;
  uint64_t* crossDeviceBarrierFlag{nullptr};

  SymmPtr interNodeChunkFlagMemObj;
  index_t* destNodeTokenCounter{nullptr};
  SymmPtr nodeRecvTokenNumMemObj;
  index_t* blockFlagCounter{nullptr};
  uint32_t* interNodeBlocksBarrier{nullptr};
  index_t* interNodeDispDestTokIdMap{nullptr};
  index_t* interNodeChunkFlagCombine{nullptr};
  index_t* interNodeDispSendMap{nullptr};
};

/* ========================================================================== */
/*  Index helpers — identical to mori/src/ops/dispatch_combine/common.hpp      */
/* ========================================================================== */
__device__ __forceinline__ int FlatTokenIndex(
    const EpDispatchCombineConfig& c, int pe, int localTokId) {
  return pe * c.MaxNumTokensToSend() + localTokId;
}
__device__ __forceinline__ int PeFromFlatTokenIndex(
    const EpDispatchCombineConfig& c, int flatIdx) {
  return flatIdx / c.MaxNumTokensToSend();
}
__device__ __forceinline__ int LocalTokIdFromFlatTokenIndex(
    const EpDispatchCombineConfig& c, int flatIdx) {
  return flatIdx % c.MaxNumTokensToSend();
}
__device__ __forceinline__ int NullFlatTokenIndex(
    const EpDispatchCombineConfig& c) {
  return c.worldSize * c.MaxNumTokensToSend();
}
__device__ __forceinline__ int SendBufSlotOffset(
    const EpDispatchCombineConfig& c, int pe, int slotId) {
  return pe * c.MaxNumTokensToSendPerRank() + slotId;
}

/* ========================================================================== */
/*  DEF_COMMON_VARS — matches mori's macro from common.hpp                    */
/* ========================================================================== */
#define DEF_COMMON_VARS                                                  \
  const EpDispatchCombineConfig& config = args.config;                   \
  const int laneId      = threadIdx.x & (kWaveSize - 1);                \
  const int warpId      = threadIdx.x / kWaveSize;                      \
  const int warpNum     = blockDim.x / kWaveSize;                       \
  const int blockNum    = gridDim.x;                                     \
  const int blockId     = blockIdx.x;                                    \
  const int globalWarpId  = blockIdx.x * warpNum + warpId;              \
  const int globalWarpNum = gridDim.x * warpNum;                        \
  const int myPe   = config.rank;                                        \
  const int myNode = myPe / config.gpuPerNode;                           \
  const int nNodes = config.worldSize / config.gpuPerNode;               \
  const int numExpertPerToken = config.numExpertPerToken;                \
  const size_t hiddenBytes = config.HiddenBytes(sizeof(T));              \
  const size_t indexBytes  = config.IndexBytes();                        \
  const size_t weightBytes = config.WeightBytes();                       \
  const size_t scaleBytes  = config.ScaleBytes();                        \
  const size_t srcTokenIdBytes = config.SrcTokenIdBytes();               \
  const size_t xferBytes   = config.XferBytesPerToken(sizeof(T));        \
  const size_t combXferBytes =                                           \
      (args.weightsBuf == nullptr) ? hiddenBytes                         \
                                   : hiddenBytes + weightBytes;          \
  (void)blockNum; (void)globalWarpNum; (void)nNodes;                     \
  (void)numExpertPerToken; (void)weightBytes; (void)combXferBytes;       \
  (void)indexBytes; (void)scaleBytes; (void)srcTokenIdBytes;             \
  (void)xferBytes;

/* ========================================================================== */
/*  WarpCopy — int4-vectorised warp copy                                      */
/* ========================================================================== */
template <typename T>
__device__ __forceinline__ void WarpCopy(
    T* __restrict__ dst, const T* __restrict__ src, size_t nelems) {
  const int lane = threadIdx.x & (kWaveSize - 1);
  constexpr int vecElems = sizeof(int4) / sizeof(T);
  const size_t nVec = nelems / vecElems;
  auto* d = reinterpret_cast<int4*>(dst);
  auto* s = reinterpret_cast<const int4*>(src);
  for (size_t i = lane; i < nVec; i += kWaveSize)
    d[i] = s[i];
  const size_t tail = nVec * vecElems;
  for (size_t i = tail + lane; i < nelems; i += kWaveSize)
    dst[i] = src[i];
}

/* ========================================================================== */
/*  WarpAccum — accumulate from multiple sources into dst (float internal)    */
/*  Mirrors mori::core::WarpAccum<T,4>(dst, srcs, nullptr, accumNum, nelems) */
/* ========================================================================== */
template <typename T>
__device__ __forceinline__ void WarpAccum(
    T* __restrict__ dst, T* const* __restrict__ srcs,
    int accumNum, size_t nelems) {
  const int lane = threadIdx.x & (kWaveSize - 1);
  constexpr int vecElems = sizeof(int4) / sizeof(T);

  for (size_t base = lane * vecElems; base + vecElems <= nelems;
       base += kWaveSize * vecElems) {
    float acc[sizeof(int4) / sizeof(T)];
    #pragma unroll
    for (int j = 0; j < vecElems; ++j) acc[j] = 0.0f;

    for (int k = 0; k < accumNum; ++k) {
      if (srcs[k] == nullptr) continue;
      int4 vec = reinterpret_cast<const int4*>(srcs[k])[base / vecElems];
      auto* v = reinterpret_cast<const T*>(&vec);
      #pragma unroll
      for (int j = 0; j < vecElems; ++j)
        acc[j] += static_cast<float>(v[j]);
    }

    int4 res;
    auto* o = reinterpret_cast<T*>(&res);
    #pragma unroll
    for (int j = 0; j < vecElems; ++j) o[j] = static_cast<T>(acc[j]);
    reinterpret_cast<int4*>(dst)[base / vecElems] = res;
  }

  const size_t tail = (nelems / vecElems) * vecElems;
  for (size_t i = tail + lane; i < nelems; i += kWaveSize) {
    float s = 0.0f;
    for (int k = 0; k < accumNum; ++k)
      if (srcs[k] != nullptr) s += static_cast<float>(srcs[k][i]);
    dst[i] = static_cast<T>(s);
  }
}

/* float specialisation: accumulate directly without fp32 conversion */
template <>
__device__ __forceinline__ void WarpAccum<float>(
    float* __restrict__ dst, float* const* __restrict__ srcs,
    int accumNum, size_t nelems) {
  const int lane = threadIdx.x & (kWaveSize - 1);
  constexpr int vecElems = sizeof(int4) / sizeof(float); // 4

  for (size_t base = lane * vecElems; base + vecElems <= nelems;
       base += kWaveSize * vecElems) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int k = 0; k < accumNum; ++k) {
      if (srcs[k] == nullptr) continue;
      int4 vec = reinterpret_cast<const int4*>(srcs[k])[base / vecElems];
      auto* v = reinterpret_cast<const float*>(&vec);
      #pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] += v[j];
    }

    int4 res;
    auto* o = reinterpret_cast<float*>(&res);
    #pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = acc[j];
    reinterpret_cast<int4*>(dst)[base / vecElems] = res;
  }

  const size_t tail = (nelems / vecElems) * vecElems;
  for (size_t i = tail + lane; i < nelems; i += kWaveSize) {
    float s = 0.f;
    for (int k = 0; k < accumNum; ++k)
      if (srcs[k] != nullptr) s += srcs[k][i];
    dst[i] = s;
  }
}

/* ========================================================================== */
/*  Kernel 1 — EpCombineSync                                                 */
/*  Copy expert outputs into shmem combineInp; copy weights into shmem.       */
/*  No RDMA. Launched with multiProcessorCount blocks.                        */
/* ========================================================================== */
template <typename T>
__global__ void EpCombineSync(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  const index_t totalRecvTokens = args.totalRecvTokenNum[0];
  const int tokenPerBlock =
      (totalRecvTokens + blockNum - 1) / blockNum;
  const int startTok = blockId * tokenPerBlock;
  const int endTok   = min(startTok + tokenPerBlock, totalRecvTokens);

  T* combInpLocal = args.interNodeV1TokBufs.combineInp.template GetAs<T*>();

  for (int t = startTok + warpId; t < endTok; t += warpNum) {
    WarpCopy(combInpLocal + t * config.hiddenDim,
             args.inpTokenBuf + t * config.hiddenDim,
             (size_t)config.hiddenDim);
  }

  if (args.weightsBuf) {
    float* wLocal =
        args.shmemInpWeightsMemObj.template GetAs<float*>();
    for (int t = startTok + warpId; t < endTok; t += warpNum) {
      WarpCopy(wLocal + t * numExpertPerToken,
               args.weightsBuf + t * numExpertPerToken,
               (size_t)numExpertPerToken);
    }
  }
}

/* ========================================================================== */
/*  Kernel 2 — EpCombineSyncBarrier                                           */
/*  Intra-node P2P barrier so every GPU's combineInp is visible.              */
/*  Launched with 1 block, kWaveSize threads.                                 */
/* ========================================================================== */
template <typename T>
__global__ void EpCombineSyncBarrier(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  uint64_t barrierFlag = 0;
  if (laneId == 0)
    barrierFlag = __hip_atomic_load(
        args.crossDeviceBarrierFlag,
        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  barrierFlag = __shfl(barrierFlag, 0);

  uint64_t* localBarrier =
      args.crossDeviceBarrierMemObj.template GetAs<uint64_t*>();

  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    uint64_t* remoteBarrier =
        args.crossDeviceBarrierMemObj.template GetAs<uint64_t*>(destPe);
    __hip_atomic_store(
        remoteBarrier + config.rank, barrierFlag,
        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    while (__hip_atomic_load(
               localBarrier + destPe,
               __ATOMIC_RELAXED,
               __HIP_MEMORY_SCOPE_SYSTEM) != barrierFlag) {
    }
  }
}

/* ========================================================================== */
/*  Kernel 3 — EpCombineInterNodeV1KernelLowLatency                          */
/*                                                                            */
/*  RDMA blocks (blockId < rdmaBlockNum):                                     */
/*    CombineInterNodeLL — for single-node the main loop is empty (nNodes-1   */
/*    == 0); only the trailing barrier bookkeeping runs.                       */
/*                                                                            */
/*  XGMI blocks (blockId >= rdmaBlockNum):                                    */
/*    CombineIntraNodeLL — gather expert outputs from peer GPUs via P2P and   */
/*    accumulate into the staging buffer.                                      */
/* ========================================================================== */
namespace v1ll_detail {

template <typename T>
__device__ void CombineIntraNodeLL(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  if (args.curRankNumToken == 0) return;

  const int blockOffset  = args.rdmaBlockNum;
  const int xgmiBlockNum = blockNum - args.rdmaBlockNum;
  const int xgmiWarpNum  = xgmiBlockNum * warpNum;

  extern __shared__ char sharedMem[];
  T**     srcPtrs     = reinterpret_cast<T**>(sharedMem)
                        + warpId * numExpertPerToken;
  float** srcWtsPtrs  = reinterpret_cast<float**>(sharedMem)
                        + warpNum * numExpertPerToken
                        + warpId * numExpertPerToken;

  uint8_t* stagingPtr =
      args.interNodeV1TokBufs.staging.template GetAs<uint8_t*>()
      + SendBufSlotOffset(config, nNodes + myNode, 0) * combXferBytes;

  const index_t warpsPerToken =
      (xgmiWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  const index_t hiddenDimPerWarp =
      (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  for (int i = globalWarpId - blockOffset * warpNum;
       i < args.curRankNumToken * warpsPerToken;
       i += xgmiWarpNum) {
    const index_t tokenId        = i / warpsPerToken;
    const index_t inTokenPartId  = i % warpsPerToken;
    const index_t hiddenOff      = inTokenPartId * hiddenDimPerWarp;
    const index_t hiddenSize     =
        max(0, min(config.hiddenDim - hiddenOff, hiddenDimPerWarp));

    if (laneId < numExpertPerToken) {
      srcPtrs[laneId]    = nullptr;
      srcWtsPtrs[laneId] = nullptr;

      index_t destTokId =
          args.dispDestTokIdMap[tokenId * numExpertPerToken + laneId];
      index_t destPe   = PeFromFlatTokenIndex(config, destTokId);
      index_t destNode = destPe / config.gpuPerNode;
      if (destNode == myNode) {
        index_t destLocalTok =
            LocalTokIdFromFlatTokenIndex(config, destTokId);
        srcPtrs[laneId] =
            args.interNodeV1TokBufs.combineInp
                .template GetAs<T*>(destPe)
            + destLocalTok * config.hiddenDim + hiddenOff;
        srcWtsPtrs[laneId] =
            args.shmemInpWeightsMemObj
                .template GetAs<float*>(destPe)
            + destLocalTok * numExpertPerToken;
      }
    }

    WarpAccum(
        reinterpret_cast<T*>(stagingPtr + tokenId * combXferBytes)
            + hiddenOff,
        srcPtrs, numExpertPerToken, (size_t)hiddenSize);

    if (args.weightsBuf && (inTokenPartId == warpsPerToken - 1)) {
      WarpAccum(
          reinterpret_cast<float*>(
              stagingPtr + tokenId * combXferBytes + hiddenBytes),
          srcWtsPtrs, numExpertPerToken, (size_t)numExpertPerToken);
    }
  }
}

template <typename T>
__device__ void CombineInterNodeLL(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  const int maxChunkNum =
      (config.MaxNumTokensToSendPerRank() + kWaveSize - 1) / kWaveSize;

  uint64_t* chunkFlag =
      args.interNodeChunkFlagMemObj.template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum =
      args.nodeRecvTokenNumMemObj.template GetAs<uint64_t*>();

  extern __shared__ char sharedMem[];
  T**     srcPtrs    = reinterpret_cast<T**>(sharedMem)
                       + warpId * numExpertPerToken;
  float** srcWtsPtrs = reinterpret_cast<float**>(sharedMem)
                       + warpNum * numExpertPerToken
                       + warpId * numExpertPerToken;
  uint8_t* stagingPtr =
      args.interNodeV1TokBufs.staging.template GetAs<uint8_t*>();

  const int rdmaWarpNum = args.rdmaBlockNum * warpNum;

  /* ------ Inter-node gather + RDMA send back ------
   * For single-node (nNodes == 1) this loop is empty. */
  for (int n = 0; n < nNodes - 1; ++n) {
    const int node = (myNode + n + 1) % nNodes;
    uint64_t nodeCount = nodeRecvTokenNum[node];
    if (nodeCount > 0) nodeCount -= 1;
    if (nodeCount == 0) continue;

    const int warpsPerToken   = 4;
    const int hiddenDimPerWarp =
        (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

    for (int i = globalWarpId;
         i < static_cast<int>(nodeCount) * warpsPerToken;
         i += rdmaWarpNum) {
      const int tokenId = i / warpsPerToken;
      const int k       = tokenId / kWaveSize;
      const int startTok = k * kWaveSize;

      uint64_t thisChunkNum = chunkFlag[node * maxChunkNum + k];
      thisChunkNum -= (thisChunkNum > 0) ? 1 : 0;
      if ((tokenId - startTok) < static_cast<int>(thisChunkNum)) {
        const int partId    = i % warpsPerToken;
        const int hidOff    = partId * hiddenDimPerWarp;
        const int hidSize   =
            max(0, min(config.hiddenDim - hidOff, hiddenDimPerWarp));

        const int gTokId =
            SendBufSlotOffset(config, node, tokenId);

        if (laneId < numExpertPerToken) {
          srcPtrs[laneId]    = nullptr;
          srcWtsPtrs[laneId] = nullptr;

          index_t destTokId = args.interNodeDispDestTokIdMap[
              gTokId * numExpertPerToken + laneId];
          index_t destPe   = PeFromFlatTokenIndex(config, destTokId);
          index_t destNode = destPe / config.gpuPerNode;
          if (destNode == myNode) {
            index_t destLocal =
                LocalTokIdFromFlatTokenIndex(config, destTokId);
            srcPtrs[laneId] =
                args.interNodeV1TokBufs.combineInp
                    .template GetAs<T*>(destPe)
                + destLocal * config.hiddenDim + hidOff;
            srcWtsPtrs[laneId] =
                args.shmemInpWeightsMemObj
                    .template GetAs<float*>(destPe)
                + destLocal * numExpertPerToken;
          }
        }

        WarpAccum(
            reinterpret_cast<T*>(
                stagingPtr + gTokId * combXferBytes) + hidOff,
            srcPtrs, numExpertPerToken, (size_t)hidSize);

        if (args.weightsBuf && (partId == 0)) {
          WarpAccum(
              reinterpret_cast<float*>(
                  stagingPtr + gTokId * combXferBytes + hiddenBytes),
              srcWtsPtrs, numExpertPerToken,
              (size_t)numExpertPerToken);
        }
      }

      index_t finished = 0;
      if (laneId == 0)
        finished = atomicAdd(
            &args.interNodeChunkFlagCombine[node * maxChunkNum + k], 1);
      finished = __shfl(finished, 0);

      if (finished + 1 >= warpsPerToken * static_cast<int>(kWaveSize)) {
        if (laneId == 0) {
          __hip_atomic_store(
              chunkFlag + node * maxChunkNum + k,
              uint64_t{0},
              __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
          __hip_atomic_store(
              args.interNodeChunkFlagCombine + node * maxChunkNum + k,
              index_t{0},
              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        const int proxyPe =
            node * config.gpuPerNode + (config.rank % config.gpuPerNode);
        const int qpId = k % config.numQpPerPe;
        (void)qpId;

        /* RDMA put combined chunk back to originating node.
         * dest/src are symmetric addresses; rocSHMEM resolves to proxyPe. */
        rocshmem_putmem_nbi_wave(
            stagingPtr
                + SendBufSlotOffset(
                      config, myNode + nNodes, startTok) * combXferBytes,
            stagingPtr
                + SendBufSlotOffset(config, node, startTok) * combXferBytes,
            thisChunkNum * combXferBytes, proxyPe);
      }
    }
  }

  /* ------ Trailing barrier (runs even when main loop is empty) ------ */
  __threadfence_system();

  int finishedWarp = 0;
  uint64_t barrierFlag = 0;
  if (laneId == 0) {
    finishedWarp = atomicAdd(
        reinterpret_cast<int*>(&args.interNodeBlocksBarrier[0]), 1);
    barrierFlag = __hip_atomic_load(
        args.crossDeviceBarrierFlag,
        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  finishedWarp = __shfl(finishedWarp, 0);
  barrierFlag  = __shfl(barrierFlag, 0);

  if (finishedWarp + 1 == args.rdmaBlockNum * warpNum) {
    if (laneId < nNodes) {
      __hip_atomic_store(
          nodeRecvTokenNum + laneId,
          uint64_t{0},
          __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    /* Remote AMO_ADD for cross-node barrier — skipped when laneId == myNode.
     * For single-node (nNodes==1), entirely skipped. */
    if (laneId < nNodes && laneId != myNode) {
      const int proxyPe =
          laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      uint64_t* barrierDest =
          args.crossDeviceBarrierMemObj.template GetAs<uint64_t*>();
      for (int q = 0; q < config.numQpPerPe; ++q) {
        rocshmem_long_atomic_add(
            reinterpret_cast<long*>(barrierDest + config.rank),
            1L, proxyPe);
      }
      __threadfence_system();
    }
    if (laneId == 0) args.interNodeBlocksBarrier[0] = 0;

    uint64_t* localBarrier =
        args.crossDeviceBarrierMemObj.template GetAs<uint64_t*>();
    if (laneId < nNodes && laneId != myNode) {
      const int proxyPe =
          laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      while (__hip_atomic_load(
                 localBarrier + proxyPe,
                 __ATOMIC_RELAXED,
                 __HIP_MEMORY_SCOPE_SYSTEM) !=
             barrierFlag * config.numQpPerPe) {
      }
    }
  }
}

}  // namespace v1ll_detail

template <typename T>
__global__ void EpCombineInterNodeV1KernelLowLatency(
    EpDispatchCombineArgs<T> args) {
  if (static_cast<int>(blockIdx.x) < args.rdmaBlockNum) {
    v1ll_detail::CombineInterNodeLL(args);
  } else {
    v1ll_detail::CombineIntraNodeLL(args);
  }
}

/* ========================================================================== */
/*  Kernel 4 — EpCombineAll                                                   */
/*  Final reduction across nodes.  For single-node (nNodes==1) this is a      */
/*  copy from the staging buffer to combineOut.                               */
/* ========================================================================== */
template <typename T>
__global__ void EpCombineAll(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  if (globalWarpId == 0) {
    if (laneId == 0) args.totalRecvTokenNum[0] = 0;
    if (laneId < nNodes) args.blockFlagCounter[laneId] = 0;
  }
  if (args.curRankNumToken == 0) return;

  extern __shared__ char sharedMem[];
  T**     srcPtrs    = reinterpret_cast<T**>(sharedMem)
                       + warpId * numExpertPerToken;
  float** srcWtsPtrs = reinterpret_cast<float**>(sharedMem)
                       + warpNum * numExpertPerToken
                       + warpId * numExpertPerToken;

  uint8_t* stagingPtr =
      args.interNodeV1TokBufs.staging.template GetAs<uint8_t*>()
      + SendBufSlotOffset(config, nNodes, 0) * combXferBytes;

  const index_t warpsPerToken =
      (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  const index_t hiddenDimPerWarp =
      (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  for (int i = globalWarpId;
       i < args.curRankNumToken * warpsPerToken;
       i += globalWarpNum) {
    const index_t tokenId       = i / warpsPerToken;
    const index_t inTokenPartId = i % warpsPerToken;
    const index_t hiddenOff     = inTokenPartId * hiddenDimPerWarp;
    const index_t hiddenSize    =
        max(0, min(config.hiddenDim - hiddenOff, hiddenDimPerWarp));

    int laneNode = -1;
    if (laneId < numExpertPerToken) {
      int lanePe =
          args.tokenIndices[tokenId * numExpertPerToken + laneId]
          / config.numExpertPerRank;
      laneNode = lanePe / config.gpuPerNode;
    }

    if (laneId < nNodes) {
      srcPtrs[laneId]    = nullptr;
      srcWtsPtrs[laneId] = nullptr;
    }

    for (int n = 0; n < nNodes; ++n) {
      if (__any(laneNode == n) && (laneId == 0)) {
        int mappedId = (n == myNode)
            ? tokenId
            : args.interNodeDispSendMap[nNodes * tokenId + n];
        uint8_t* base =
            stagingPtr
            + SendBufSlotOffset(config, n, mappedId) * combXferBytes;
        srcPtrs[n]    = reinterpret_cast<T*>(base) + hiddenOff;
        srcWtsPtrs[n] = reinterpret_cast<float*>(base + hiddenBytes);
      }
    }

    T* combOutPtr =
        args.interNodeV1TokBufs.combineOut.template GetAs<T*>();
    WarpAccum(
        combOutPtr + tokenId * config.hiddenDim + hiddenOff,
        srcPtrs, nNodes, (size_t)hiddenSize);

    if (args.weightsBuf && (inTokenPartId == warpsPerToken - 1)) {
      float* combOutWts =
          args.shmemCombineOutWeightsMemObj.template GetAs<float*>();
      WarpAccum(
          combOutWts + tokenId * numExpertPerToken,
          srcWtsPtrs, nNodes, (size_t)numExpertPerToken);
    }
  }
}

/* ========================================================================== */
/*  Host launcher — fires the 4 kernels in sequence on the given stream       */
/* ========================================================================== */
template <typename T>
void LaunchCombineV1LL(
    EpDispatchCombineArgs<T>& args,
    int blockNum, int warpPerBlock, int rdmaBlockNum,
    int multiProcessorCount, hipStream_t stream) {

  const unsigned int blockX = kWaveSize * warpPerBlock;
  const int smemCombine =
      warpPerBlock * args.config.numExpertPerToken
      * static_cast<int>(sizeof(T*) + sizeof(float*));

  EpCombineSync<T>
      <<<multiProcessorCount, blockX, 0, stream>>>(args);

  EpCombineSyncBarrier<T>
      <<<1, kWaveSize, 0, stream>>>(args);

  EpCombineInterNodeV1KernelLowLatency<T>
      <<<blockNum, blockX, smemCombine, stream>>>(args);

  EpCombineAll<T>
      <<<multiProcessorCount, blockX, smemCombine, stream>>>(args);
}

}  // namespace mori_v1ll
