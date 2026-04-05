/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Port of mori's InterNodeV1LL dispatch kernels to direct rocSHMEM APIs.
 * Maintains the exact same data layout, struct definitions, and algorithmic
 * logic as mori's EpDispatchInterNodeV1KernelLowLatency path.
 *
 * Depends on LL_MoE_Combine_V1LL.hpp for shared types (EpDispatchCombineConfig,
 * EpDispatchCombineArgs, SymmPtr, ShmemBufsInterNodeV1, helpers, WarpCopy).
 *
 *****************************************************************************/

#pragma once

#include "LL_MoE_Combine_V1LL.hpp"

namespace mori_v1ll {

/* ========================================================================== */
/*  Device helpers for dispatch                                               */
/* ========================================================================== */

__device__ __forceinline__ int CeilDiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ index_t AtomicLoadRelaxedSystem(index_t* addr) {
  return __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint64_t AtomicLoadRelaxedSystem(uint64_t* addr) {
  return __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ index_t AtomicLoadRelaxed(index_t* addr) {
  return __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ uint64_t AtomicLoadRelaxed(uint64_t* addr) {
  return __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <typename U>
__device__ __forceinline__ void AtomicStoreRelaxedSystem(U* addr, U val) {
  __hip_atomic_store(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename U>
__device__ __forceinline__ void AtomicStoreSeqCstSystem(U* addr, U val) {
  __hip_atomic_store(addr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename U>
__device__ __forceinline__ U AtomicLoadSeqCstSystem(U* addr) {
  return __hip_atomic_load(addr, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ index_t Int32WaitUntilGreaterThan(index_t* addr, index_t val) {
  index_t cur;
  while ((cur = __hip_atomic_load(addr, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM)) <= val) {}
  return cur;
}

/* ========================================================================== */
/*  Kernel 1 — EpDispatchCopyToStaging                                       */
/*  Packs input tokens into the staging buffer in the interleaved layout:     */
/*    [hidden | indices | weights | scales | srcTokId]                        */
/*  Launched with multiProcessorCount blocks.                                */
/* ========================================================================== */
template <typename T>
__global__ void EpDispatchCopyToStaging(EpDispatchCombineArgs<T> args) {
  DEF_COMMON_VARS;

  if (args.curRankNumToken == 0) return;

  index_t warpsPerToken = (globalWarpNum + args.curRankNumToken - 1) / args.curRankNumToken;
  index_t hiddenDimPerWarp = (config.hiddenDim + warpsPerToken - 1) / warpsPerToken;

  uint8_t* stagingPtr =
      args.interNodeV1TokBufs.staging.template GetAs<uint8_t*>();

  for (int i = globalWarpId; i < args.curRankNumToken * warpsPerToken;
       i += globalWarpNum) {
    index_t tokenId       = i / warpsPerToken;
    index_t inTokenPartId = i % warpsPerToken;
    index_t hiddenOff     = inTokenPartId * hiddenDimPerWarp;
    index_t hiddenSize    = max(0, min(config.hiddenDim - hiddenOff, hiddenDimPerWarp));

    size_t stagingTokOff = tokenId * xferBytes;

    WarpCopy(reinterpret_cast<uint8_t*>(stagingPtr + stagingTokOff + hiddenOff * sizeof(T)),
             reinterpret_cast<uint8_t*>(args.inpTokenBuf) +
                 tokenId * hiddenBytes + hiddenOff * sizeof(T),
             (size_t)(hiddenSize * sizeof(T)));

    if (inTokenPartId != 0) continue;

    WarpCopy(reinterpret_cast<uint8_t*>(stagingPtr + stagingTokOff + hiddenBytes),
             reinterpret_cast<uint8_t*>(args.tokenIndices) + tokenId * indexBytes,
             indexBytes);

    WarpCopy(reinterpret_cast<uint8_t*>(stagingPtr + stagingTokOff + hiddenBytes + indexBytes),
             reinterpret_cast<uint8_t*>(args.weightsBuf) + tokenId * weightBytes,
             weightBytes);

    if (args.scalesBuf && (scaleBytes > 0))
      WarpCopy(stagingPtr + stagingTokOff + hiddenBytes + indexBytes + weightBytes,
               args.scalesBuf + tokenId * scaleBytes,
               scaleBytes);

    if (laneId == 0)
      reinterpret_cast<index_t*>(
          stagingPtr + stagingTokOff + hiddenBytes + indexBytes +
          weightBytes + scaleBytes)[0] =
          static_cast<index_t>(FlatTokenIndex(config, config.rank, tokenId));
  }
}

/* ========================================================================== */
/*  Dispatch sub-kernels (in v1ll_dispatch namespace)                         */
/* ========================================================================== */
namespace v1ll_dispatch {

/* ---------------------------------------------------------------------- */
/*  DispatchIntraNodeBlock — P2P copy one (token, expert) to dest PE      */
/* ---------------------------------------------------------------------- */
template <typename T>
__device__ void DispatchIntraNodeBlock(
    EpDispatchCombineArgs<T>& args, int tokenId, int expId,
    int destPe, int& localPeTokenCounter) {
  DEF_COMMON_VARS;

  index_t tokenExpertId = tokenId * config.numExpertPerToken + expId;
  index_t destTokId = 0;
  if (laneId == 0) {
    destTokId = atomicAdd(
        args.dispTokOffsetMemObj.template GetAs<index_t*>(destPe), 1);
    args.dispDestTokIdMap[tokenExpertId] =
        FlatTokenIndex(config, destPe, destTokId);
    AtomicStoreRelaxedSystem(
        args.dispTokIdToSrcTokIdMemObj.template GetAs<index_t*>(destPe) + destTokId,
        static_cast<index_t>(FlatTokenIndex(config, config.rank, tokenId)));
  }
  if (laneId == (destPe % config.gpuPerNode)) localPeTokenCounter++;
  destTokId = __shfl(destTokId, 0);
  size_t srcTokOff  = tokenId  * config.hiddenDim;
  size_t destTokOff = destTokId * config.hiddenDim;

  T* remoteToken = args.interNodeV1TokBufs.dispatchOut.template GetAs<T*>(destPe);
  const T* localToken = args.inpTokenBuf;
  WarpCopy(remoteToken + destTokOff, localToken + srcTokOff, (size_t)config.hiddenDim);

  index_t* remoteIdx = args.shmemOutIndicesMemObj.template GetAs<index_t*>(destPe);
  const index_t* localIdx = args.tokenIndices;
  WarpCopy(remoteIdx + destTokId * config.numExpertPerToken,
           localIdx + tokenId * config.numExpertPerToken,
           (size_t)config.numExpertPerToken);

  float* remoteWt = args.shmemDispatchOutWeightsMemObj.template GetAs<float*>(destPe);
  const float* localWt = args.weightsBuf;
  WarpCopy(remoteWt + destTokId * config.numExpertPerToken,
           localWt + tokenId * config.numExpertPerToken,
           (size_t)config.numExpertPerToken);

  if (args.scalesBuf && (scaleBytes > 0)) {
    WarpCopy(args.shmemOutScalesMemObj.template GetAs<uint8_t*>(destPe) +
                 destTokId * scaleBytes,
             args.scalesBuf + tokenId * scaleBytes,
             scaleBytes);
  }
}

/* ---------------------------------------------------------------------- */
/*  DispatchIntraNode — XGMI blocks dispatch tokens to intra-node peers   */
/* ---------------------------------------------------------------------- */
template <typename T>
__device__ void DispatchIntraNode(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  int blockOffset   = args.rdmaBlockNum;
  int xgmiBlockNum  = blockNum - args.rdmaBlockNum;
  int tokenPerBlock = (args.curRankNumToken + xgmiBlockNum - 1) / xgmiBlockNum;
  int startTok      = (blockId - blockOffset) * tokenPerBlock;
  int endTok        = min(startTok + tokenPerBlock, args.curRankNumToken);

  int localPeTokenCounter = 0;

  for (int i = warpId; i < (endTok - startTok) * config.numExpertPerToken;
       i += warpNum) {
    index_t tokenId = i / config.numExpertPerToken + startTok;
    index_t destPe  = args.tokenIndices[startTok * config.numExpertPerToken + i]
                      / config.numExpertPerRank;
    int destNode = destPe / config.gpuPerNode;

    int lanePe = -1, laneNode = -1;
    if (laneId < numExpertPerToken) {
      lanePe   = args.tokenIndices[tokenId * numExpertPerToken + laneId]
                 / config.numExpertPerRank;
      laneNode = lanePe / config.gpuPerNode;
    }

    index_t inTokenExpertId = i % numExpertPerToken;
    if (destNode == myNode) {
      if (__any((laneId < inTokenExpertId) && (destPe == lanePe))) {
        if (laneId == 0)
          args.dispDestTokIdMap[startTok * config.numExpertPerToken + i] =
              NullFlatTokenIndex(config);
        continue;
      }
      DispatchIntraNodeBlock(args, tokenId, inTokenExpertId, destPe,
                             localPeTokenCounter);
    }
  }

  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    atomicAdd(args.destPeTokenCounter + destPe, localPeTokenCounter);
  }
}

/* ---------------------------------------------------------------------- */
/*  DispatchInterNodeLLSend — RDMA blocks send tokens to remote nodes     */
/*  For single node (nNodes==1), the outer loop is empty.                 */
/* ---------------------------------------------------------------------- */
template <typename T>
__device__ void DispatchInterNodeLLSend(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  int maxChunkNum   = CeilDiv(config.MaxNumTokensToSendPerRank(), (int)kWaveSize);
  int totalChunkNum = CeilDiv(args.curRankNumToken, (int)kWaveSize);
  int blockChunkNum = CeilDiv(totalChunkNum, args.rdmaBlockNum);

  int chunkStartTok = blockChunkNum * blockId * kWaveSize;
  int chunkEndTok   = min(chunkStartTok + blockChunkNum * (int)kWaveSize,
                          args.curRankNumToken);

  uint8_t* stagingPtr =
      args.interNodeV1TokBufs.staging.template GetAs<uint8_t*>();

  for (int i = warpId; i < nNodes; i += warpNum) {
    if (i == myNode) continue;
    int proxyPe = i * config.gpuPerNode + (config.rank % config.gpuPerNode);

    for (int tokenId = chunkStartTok + laneId; tokenId < chunkEndTok;
         tokenId += kWaveSize) {
      bool shouldSend = false;
      for (int e = 0; e < config.numExpertPerToken; e++) {
        int destNode = args.tokenIndices[tokenId * numExpertPerToken + e]
                       / config.numExpertPerRank / config.gpuPerNode;
        if (destNode == i) {
          shouldSend = true;
          args.dispDestTokIdMap[tokenId * numExpertPerToken + e] =
              NullFlatTokenIndex(config);
        }
      }

      index_t flagSlotId = 0;
      if (laneId == 0)
        flagSlotId = atomicAdd(args.blockFlagCounter + i, 1);
      flagSlotId = __shfl(flagSlotId, 0);

      index_t destTokIdOff = flagSlotId * kWaveSize;
      index_t destTokId    = destTokIdOff + laneId;

      size_t remoteIdx = SendBufSlotOffset(config, myNode, destTokId);
      if (laneId == 0) {
        index_t tokenNum = min(tokenId + (int)kWaveSize, chunkEndTok) - tokenId;
        size_t stagingOff = tokenId * xferBytes;
        uint8_t* dispInpPtr =
            args.interNodeV1TokBufs.dispatchInp.template GetAs<uint8_t*>();

        rocshmem_putmem_nbi_wave(
            dispInpPtr + remoteIdx * xferBytes,
            stagingPtr + stagingOff,
            tokenNum * xferBytes, proxyPe);

        uint64_t* chunkFlagPtr =
            args.interNodeChunkFlagMemObj.template GetAs<uint64_t*>();
        rocshmem_long_atomic_add(
            reinterpret_cast<long*>(chunkFlagPtr + myNode * maxChunkNum + flagSlotId),
            (long)(tokenNum + 1), proxyPe);
      }
      if (shouldSend)
        args.interNodeDispSendMap[nNodes * tokenId + i] = destTokId;
    }
  }

  int finishedWarp = 0;
  if (laneId == 0)
    finishedWarp = atomicAdd(&args.interNodeBlocksBarrier[1], 1);
  finishedWarp = __shfl(finishedWarp, 0);

  if (finishedWarp + 1 == args.rdmaBlockNum * warpNum) {
    if (laneId < nNodes) {
      int proxyPe = laneId * config.gpuPerNode + (config.rank % config.gpuPerNode);
      index_t numTokenSignal =
          AtomicLoadRelaxed(args.blockFlagCounter + laneId) * kWaveSize + 1;
      uint64_t* nodeRecvPtr =
          args.nodeRecvTokenNumMemObj.template GetAs<uint64_t*>();
      rocshmem_long_atomic_add(
          reinterpret_cast<long*>(nodeRecvPtr + myNode),
          (long)numTokenSignal, proxyPe);
    }
    if (laneId == 0) args.interNodeBlocksBarrier[1] = 0;
  }
}

/* ---------------------------------------------------------------------- */
/*  DispatchInterNodeLLRecv — RDMA blocks unpack from remote nodes        */
/*  For single node (nNodes==1), the outer loop is empty.                 */
/* ---------------------------------------------------------------------- */
template <typename T>
__device__ void DispatchInterNodeLLRecv(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  int maxChunkNum = CeilDiv(config.MaxNumTokensToSendPerRank(), (int)kWaveSize);

  uint64_t* chunkFlag =
      args.interNodeChunkFlagMemObj.template GetAs<uint64_t*>();
  uint64_t* nodeRecvTokenNum =
      args.nodeRecvTokenNumMemObj.template GetAs<uint64_t*>();
  uint8_t* stagingPtr =
      args.interNodeV1TokBufs.dispatchInp.template GetAs<uint8_t*>();

  int localPeTokenCounter = 0;

  for (int i = globalWarpId;
       i < config.MaxNumTokensToSendPerRank() * config.numExpertPerToken * (nNodes - 1);
       i += args.rdmaBlockNum * warpNum) {
    int expertId = i % config.numExpertPerToken;
    int tokenId  = i / config.numExpertPerToken % config.MaxNumTokensToSendPerRank();
    int nodeId   = i / config.numExpertPerToken / config.MaxNumTokensToSendPerRank();

    int node = (myNode + 1 + nodeId) % nNodes;
    int k = tokenId / kWaveSize;
    int startTok = k * kWaveSize;

    uint64_t thisChunkNum = 0;
    index_t nodeFlag = 0;
    if (laneId == 0) {
      while (1) {
        thisChunkNum = AtomicLoadRelaxedSystem(&chunkFlag[node * maxChunkNum + k]);
        if (thisChunkNum > 0) break;
        nodeFlag = (index_t)AtomicLoadRelaxedSystem(&nodeRecvTokenNum[node]);
        if ((nodeFlag > 0) && (startTok >= (nodeFlag - 1))) {
          thisChunkNum = 1;
          break;
        }
      }
    }
    thisChunkNum = __shfl(thisChunkNum, 0) - 1;
    int endTok = startTok + (int)thisChunkNum;
    if (tokenId >= endTok) continue;

    int gTokId = SendBufSlotOffset(config, node, tokenId);
    index_t* indices =
        reinterpret_cast<index_t*>(stagingPtr + gTokId * xferBytes + hiddenBytes);
    int lanePe = -1;
    if (laneId < config.numExpertPerToken)
      lanePe = indices[laneId] / config.numExpertPerRank;

    index_t srcTokId =
        reinterpret_cast<index_t*>(
            stagingPtr + gTokId * xferBytes + hiddenBytes +
            indexBytes + weightBytes + scaleBytes)[0];

    int destPe   = __shfl(lanePe, expertId);
    int destNode = destPe / config.gpuPerNode;
    bool shouldSkip = (destNode != myNode) ||
                      __any((laneId < expertId) && (destPe == lanePe));
    if (shouldSkip) {
      if (laneId == 0)
        args.interNodeDispDestTokIdMap[gTokId * config.numExpertPerToken + expertId] =
            NullFlatTokenIndex(config);
      continue;
    }

    int destTokId = 0;
    if (laneId == 0) {
      destTokId = atomicAdd(
          args.dispTokOffsetMemObj.template GetAs<index_t*>(destPe), 1);
      args.interNodeDispDestTokIdMap[gTokId * config.numExpertPerToken + expertId] =
          FlatTokenIndex(config, destPe, destTokId);
      args.dispTokIdToSrcTokIdMemObj.template GetAs<index_t*>(destPe)[destTokId] =
          srcTokId;
    }
    if ((destPe % config.gpuPerNode) == laneId) localPeTokenCounter++;
    destTokId = __shfl(destTokId, 0);

    WarpCopy(args.interNodeV1TokBufs.dispatchOut.template GetAs<uint8_t*>(destPe) +
                 destTokId * hiddenBytes,
             stagingPtr + gTokId * xferBytes,
             hiddenBytes);
    WarpCopy(args.shmemOutIndicesMemObj.template GetAs<uint8_t*>(destPe) +
                 destTokId * indexBytes,
             stagingPtr + gTokId * xferBytes + hiddenBytes,
             indexBytes);
    WarpCopy(args.shmemDispatchOutWeightsMemObj.template GetAs<uint8_t*>(destPe) +
                 destTokId * weightBytes,
             stagingPtr + gTokId * xferBytes + hiddenBytes + indexBytes,
             weightBytes);
    if (scaleBytes > 0) {
      WarpCopy(args.shmemOutScalesMemObj.template GetAs<uint8_t*>(destPe) +
                   destTokId * scaleBytes,
               stagingPtr + gTokId * xferBytes + hiddenBytes + indexBytes + weightBytes,
               scaleBytes);
    }
  }

  if (laneId < config.gpuPerNode) {
    int destPe = myNode * config.gpuPerNode + laneId;
    atomicAdd(args.destPeTokenCounter + destPe, localPeTokenCounter);
  }
}

/* ---------------------------------------------------------------------- */
/*  DispatchSync — grid barrier + signal recv token counts to peers       */
/* ---------------------------------------------------------------------- */
template <typename T>
__device__ void DispatchSync(EpDispatchCombineArgs<T>& args) {
  DEF_COMMON_VARS;

  int nodePeOffset = myNode * config.gpuPerNode;

  int finishedWarp = 0;
  if (laneId == 0) finishedWarp = atomicAdd(args.dispatchGridBarrier, 1);
  finishedWarp = __shfl(finishedWarp, 0);

  if (finishedWarp + 1 == globalWarpNum) {
    if (laneId < config.gpuPerNode) {
      int destPe = myNode * config.gpuPerNode + laneId;
      index_t numTokenSignal =
          AtomicLoadSeqCstSystem(args.destPeTokenCounter + destPe) + 1;
      index_t* signal =
          args.recvTokenNumMemObj.template GetAs<index_t*>(destPe) + myPe;
      AtomicStoreSeqCstSystem(signal, numTokenSignal);
    }
    if (laneId == 0) args.dispatchGridBarrier[0] = 0;

    index_t* recvTokenNums =
        args.recvTokenNumMemObj.template GetAs<index_t*>();
    for (int destPe = nodePeOffset + laneId;
         destPe < nodePeOffset + config.gpuPerNode;
         destPe += kWaveSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = Int32WaitUntilGreaterThan(signal, 0) - 1;
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      __threadfence_system();
      AtomicStoreSeqCstSystem(signal, (index_t)0);
      AtomicStoreSeqCstSystem(args.destPeTokenCounter + destPe, (index_t)0);
    }

    if (laneId == 0) {
      args.dispTokOffsetMemObj.template GetAs<index_t*>()[0] = 0;
      atomicAdd(args.crossDeviceBarrierFlag, 1ULL);
      args.combineGridBarrier[1] = 0;
    }
  }
}

}  // namespace v1ll_dispatch

/* ========================================================================== */
/*  Kernel 2 — EpDispatchInterNodeV1KernelLowLatency                        */
/*                                                                            */
/*  RDMA blocks (blockId < rdmaBlockNum):                                     */
/*    DispatchInterNodeLLSend + DispatchInterNodeLLRecv                       */
/*    For single-node both loops are empty.                                   */
/*                                                                            */
/*  XGMI blocks (blockId >= rdmaBlockNum):                                    */
/*    DispatchIntraNode — P2P dispatch to intra-node peers                    */
/*                                                                            */
/*  Then all blocks: DispatchSync                                             */
/* ========================================================================== */
template <typename T>
__global__ void EpDispatchInterNodeV1KernelLowLatency(
    EpDispatchCombineArgs<T> args) {
  if (static_cast<int>(blockIdx.x) < args.rdmaBlockNum) {
    v1ll_dispatch::DispatchInterNodeLLSend(args);
    v1ll_dispatch::DispatchInterNodeLLRecv(args);
  } else {
    v1ll_dispatch::DispatchIntraNode(args);
  }
  v1ll_dispatch::DispatchSync(args);
}

/* ========================================================================== */
/*  Host launcher — fires the 2 dispatch kernels in sequence                 */
/* ========================================================================== */
template <typename T>
void LaunchDispatchV1LL(
    EpDispatchCombineArgs<T>& args,
    int blockNum, int warpPerBlock, int rdmaBlockNum,
    int multiProcessorCount, hipStream_t stream) {

  const unsigned int blockX = kWaveSize * warpPerBlock;

  EpDispatchCopyToStaging<T>
      <<<multiProcessorCount, blockX, 0, stream>>>(args);

  EpDispatchInterNodeV1KernelLowLatency<T>
      <<<blockNum, blockX, 0, stream>>>(args);
}

}  // namespace mori_v1ll
