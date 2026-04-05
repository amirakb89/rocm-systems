/*
 * Pybind11 module for the V1LL dispatch + combine kernel wrapper.
 *
 * Exposes a V1LLBuffer class that owns all symmetric and device allocations,
 * and provides dispatch() / combine() methods taking/returning torch::Tensors.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <rocshmem/rocshmem.hpp>

#include "LL_MoE_Dispatch_V1LL.hpp"
#include "LL_MoE_Combine_V1LL.hpp"
#include "v1ll_api.hpp"

namespace py = pybind11;
using namespace mori_v1ll;

#define HIP_CHECK(call) do {                                     \
  hipError_t e = (call);                                         \
  if (e != hipSuccess) throw std::runtime_error(                 \
      std::string("HIP error: ") + hipGetErrorString(e));        \
} while (0)

/* ================================================================== */
/*  V1LLBuffer — owns all allocations, exposes dispatch / combine     */
/* ================================================================== */
struct V1LLBuffer {
  int rank, worldSize, gpuPerNode, nNodes;
  int hiddenDim, numExpertPerToken, numExpertPerRank;
  int maxNumInpTokenPerRank;
  int warpPerBlock, blockNum, rdmaBlockNum;
  int multiProcessorCount;
  bool initialized = false;

  EpDispatchCombineConfig cfg{};
  EpDispatchCombineArgs<hip_bfloat16> args{};
  hipStream_t stream = nullptr;

  std::vector<void*> shmem_ptrs;
  std::vector<void*> device_ptrs;

  V1LLBuffer(int rank_, int world_size_,
             int hidden_dim_, int num_expert_per_token_,
             int num_expert_per_rank_, int max_tokens_per_rank_,
             int warp_per_block_, int block_num_, int rdma_block_num_)
      : rank(rank_), worldSize(world_size_),
        hiddenDim(hidden_dim_), numExpertPerToken(num_expert_per_token_),
        numExpertPerRank(num_expert_per_rank_),
        maxNumInpTokenPerRank(max_tokens_per_rank_),
        warpPerBlock(warp_per_block_), blockNum(block_num_),
        rdmaBlockNum(rdma_block_num_) {
    gpuPerNode = worldSize;
    nNodes = 1;

    int devCount;
    HIP_CHECK(hipGetDeviceCount(&devCount));
    int devId = rank % devCount;
    HIP_CHECK(hipSetDevice(devId));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, devId));
    multiProcessorCount = prop.multiProcessorCount;

    stream = at::cuda::getCurrentCUDAStream().stream();

    build_config();
    allocate_buffers();
    initialized = true;
  }

  ~V1LLBuffer() {
    if (!initialized) return;
    free_buffers();
  }

  void build_config() {
    auto& c            = cfg;
    c.rank             = rank;
    c.worldSize        = worldSize;
    c.hiddenDim        = hiddenDim;
    c.scaleDim         = 0;
    c.scaleTypeSize    = 0;
    c.maxTokenTypeSize = (int)sizeof(hip_bfloat16);
    c.maxNumInpTokenPerRank = maxNumInpTokenPerRank;
    c.numExpertPerRank = numExpertPerRank;
    c.numExpertPerToken = numExpertPerToken;
    c.maxTotalRecvTokens = 0;
    c.warpNumPerBlock  = warpPerBlock;
    c.blockNum         = blockNum;
    c.useExternalInpBuffer = true;
    c.gpuPerNode       = gpuPerNode;
    c.rdmaBlockNum     = rdmaBlockNum;
    c.numQpPerPe       = 1;
  }

  SymmPtr salloc(size_t bytes) {
    SymmPtr p;
    if (bytes == 0) return p;
    p.localPtr = v1ll_rt::shmem_alloc(bytes);
    if (!p.localPtr) throw std::runtime_error("rocshmem_malloc failed");
    HIP_CHECK(hipMemsetAsync(p.localPtr, 0, bytes, stream));
    shmem_ptrs.push_back(p.localPtr);
    return p;
  }

  template <typename U>
  U* dalloc(size_t count) {
    U* ptr = nullptr;
    if (count == 0) return ptr;
    HIP_CHECK(hipMalloc(&ptr, count * sizeof(U)));
    HIP_CHECK(hipMemsetAsync(ptr, 0, count * sizeof(U), stream));
    device_ptrs.push_back(ptr);
    return ptr;
  }

  void allocate_buffers() {
    const auto& c = cfg;
    const size_t typeSize = sizeof(hip_bfloat16);
    const size_t xferBPT  = c.XferBytesPerToken(typeSize);
    const size_t combBPT  = c.HiddenBytes(typeSize) + c.WeightBytes();

    const size_t dispOutSz    = (size_t)c.MaxNumTokensToRecv() * c.hiddenDim * typeSize;
    const size_t dispInpSz    = (size_t)nNodes * c.MaxNumTokensToSendPerRank() * xferBPT;
    const size_t combInpSz    = (size_t)c.MaxNumTokensToRecv() * combBPT;
    const size_t stagingSz    = (size_t)c.MaxNumTokensToSendPerRank() * xferBPT;
    const size_t combStageSz  = (size_t)2 * nNodes * c.MaxNumTokensToSendPerRank() * combBPT;
    const size_t combOutSz    = (size_t)c.MaxNumTokensToSendPerRank() * c.hiddenDim * typeSize;

    args.interNodeV1TokBufs.dispatchInp = salloc(dispInpSz);
    args.interNodeV1TokBufs.combineInp  = salloc(combInpSz);
    args.interNodeV1TokBufs.staging     = salloc(std::max(stagingSz, combStageSz));
    args.interNodeV1TokBufs.dispatchOut = salloc(dispOutSz);
    args.interNodeV1TokBufs.combineOut  = salloc(combOutSz);

    const size_t maxWtSz = (size_t)c.MaxNumTokensToRecv() * c.numExpertPerToken * sizeof(float);
    args.shmemInpWeightsMemObj         = salloc(maxWtSz);
    args.shmemDispatchOutWeightsMemObj = salloc(maxWtSz);
    args.shmemCombineOutWeightsMemObj  = salloc(maxWtSz);

    const size_t maxIdxSz = (size_t)c.MaxNumTokensToRecv() * c.numExpertPerToken * sizeof(index_t);
    args.shmemInpIndicesMemObj = salloc(maxIdxSz);
    args.shmemOutIndicesMemObj = salloc(maxIdxSz);

    const size_t sigSz = (size_t)c.worldSize * sizeof(index_t) * 2;
    args.recvTokenNumMemObj      = salloc(sigSz);
    args.sendTokenNumMemObj      = salloc(sigSz);
    args.sendAtomicSignalMemObj  = salloc((size_t)(c.worldSize * 2) * sizeof(int64_t) * 2);

    args.nodeRecvTokenNumMemObj = salloc((size_t)nNodes * sizeof(uint64_t));
    args.dispTokOffsetMemObj    = salloc(sizeof(index_t));
    const size_t maxOutTok = (size_t)c.MaxNumTokensToSend() * c.numExpertPerRank;
    args.dispTokIdToSrcTokIdMemObj = salloc(maxOutTok * sizeof(index_t));
    args.crossDeviceBarrierMemObj  = salloc((size_t)c.worldSize * 2 * sizeof(uint64_t));
    args.interNodeChunkFlagMemObj  = salloc(
        (size_t)nNodes * c.MaxNumTokensToSendPerRank() * sizeof(uint64_t));

    args.inpTokenBuf   = nullptr;
    args.weightsBuf    = nullptr;
    args.tokenIndices  = nullptr;
    args.scalesBuf     = nullptr;

    args.dispatchGridBarrier  = dalloc<uint32_t>(c.worldSize);
    args.combineGridBarrier   = dalloc<uint32_t>(c.worldSize);
    args.destPeTokenCounter   = dalloc<index_t>(c.worldSize);
    args.localPeTokenCounter  = dalloc<index_t>(c.worldSize);
    args.dispReceiverIdxMap   = dalloc<index_t>(maxOutTok);
    args.dispSenderIdxMap     = dalloc<index_t>(maxOutTok);
    args.destPeTokenIdxMap    = dalloc<index_t>(maxOutTok);
    args.srcPeTokenIdxMap     = dalloc<index_t>(maxOutTok);
    args.dispDestTokIdMap     = dalloc<index_t>(maxOutTok);
    args.totalRecvTokenNum    = dalloc<index_t>(1);
    args.crossDeviceBarrierFlag = dalloc<uint64_t>(1);
    args.blockFlagCounter     = dalloc<index_t>(nNodes);
    args.interNodeBlocksBarrier = dalloc<uint32_t>(4);

    const size_t interTokMap =
        (size_t)nNodes * c.MaxNumTokensToSendPerRank() * c.numExpertPerToken;
    args.interNodeDispDestTokIdMap = dalloc<index_t>(interTokMap);
    args.interNodeChunkFlagCombine =
        dalloc<index_t>((size_t)nNodes * c.MaxNumTokensToSendPerRank());
    args.interNodeDispSendMap =
        dalloc<index_t>((size_t)nNodes * c.MaxNumTokensToSendPerRank());
    args.destNodeTokenCounter = dalloc<index_t>(nNodes);

    args.config      = cfg;
    args.rdmaBlockNum = rdmaBlockNum;

    uint64_t one = 1;
    HIP_CHECK(hipMemcpy(args.crossDeviceBarrierFlag, &one,
                         sizeof(uint64_t), hipMemcpyHostToDevice));

    HIP_CHECK(hipStreamSynchronize(stream));
  }

  void free_buffers() {
    for (auto* p : shmem_ptrs) v1ll_rt::shmem_free_ptr(p);
    shmem_ptrs.clear();
    for (auto* p : device_ptrs) (void)hipFree(p);
    device_ptrs.clear();
  }

  void reset_counters() {
    auto& c = cfg;
    HIP_CHECK(hipMemsetAsync(args.totalRecvTokenNum, 0, sizeof(index_t), stream));
    HIP_CHECK(hipMemsetAsync(args.destPeTokenCounter, 0,
                              c.worldSize * sizeof(index_t), stream));
    HIP_CHECK(hipMemsetAsync(args.dispatchGridBarrier, 0,
                              c.worldSize * sizeof(uint32_t), stream));
    HIP_CHECK(hipMemsetAsync(args.combineGridBarrier, 0,
                              c.worldSize * sizeof(uint32_t), stream));
    HIP_CHECK(hipMemsetAsync(args.blockFlagCounter, 0,
                              nNodes * sizeof(index_t), stream));
    HIP_CHECK(hipMemsetAsync(args.interNodeBlocksBarrier, 0,
                              4 * sizeof(uint32_t), stream));
    size_t maxOutTok = (size_t)c.MaxNumTokensToSend() * c.numExpertPerRank;
    HIP_CHECK(hipMemsetAsync(args.dispDestTokIdMap, 0,
                              maxOutTok * sizeof(index_t), stream));
    HIP_CHECK(hipMemsetAsync(args.dispTokOffsetMemObj.localPtr, 0,
                              sizeof(index_t), stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    v1ll_rt::shmem_barrier();
  }

  /* ---------------------------------------------------------------- */
  /*  dispatch(x, topk_idx, topk_weights) -> (recv_x, recv_indices,   */
  /*           recv_weights, num_recv, src_tok_id_map)                 */
  /* ---------------------------------------------------------------- */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor, torch::Tensor>
  dispatch(const torch::Tensor& x,
           const torch::Tensor& topk_idx,
           const torch::Tensor& topk_weights) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(topk_idx.is_cuda() && topk_idx.dtype() == torch::kInt32);
    TORCH_CHECK(topk_weights.is_cuda() && topk_weights.dtype() == torch::kFloat32);

    int numTokens = x.size(0);
    TORCH_CHECK(x.size(1) == hiddenDim);
    TORCH_CHECK(topk_idx.size(0) == numTokens && topk_idx.size(1) == numExpertPerToken);

    stream = at::cuda::getCurrentCUDAStream().stream();

    args.curRankNumToken = numTokens;
    args.inpTokenBuf  = reinterpret_cast<hip_bfloat16*>(x.data_ptr());
    args.tokenIndices = reinterpret_cast<index_t*>(topk_idx.data_ptr());
    args.weightsBuf   = reinterpret_cast<float*>(topk_weights.data_ptr());
    args.scalesBuf    = nullptr;

    v1ll_rt::launch_dispatch(args, blockNum, warpPerBlock,
                             rdmaBlockNum, multiProcessorCount, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    index_t h_totalRecv = 0;
    HIP_CHECK(hipMemcpy(&h_totalRecv, args.totalRecvTokenNum,
                         sizeof(index_t), hipMemcpyDeviceToHost));

    const int maxRecv = cfg.MaxNumTokensToRecv();
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(x.device());
    auto opts_i32  = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto opts_f32  = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    auto recv_x = torch::from_blob(
        args.interNodeV1TokBufs.dispatchOut.localPtr,
        {maxRecv, hiddenDim}, opts_bf16).slice(0, 0, h_totalRecv).clone();

    auto recv_indices = torch::from_blob(
        args.shmemOutIndicesMemObj.localPtr,
        {maxRecv, numExpertPerToken}, opts_i32).slice(0, 0, h_totalRecv).clone();

    auto recv_weights = torch::from_blob(
        args.shmemDispatchOutWeightsMemObj.localPtr,
        {maxRecv, numExpertPerToken}, opts_f32).slice(0, 0, h_totalRecv).clone();

    auto num_recv = torch::tensor({h_totalRecv}, opts_i32);

    auto src_tok_map = torch::from_blob(
        args.dispTokIdToSrcTokIdMemObj.localPtr,
        {maxRecv}, opts_i32).slice(0, 0, h_totalRecv).clone();

    return {recv_x, recv_indices, recv_weights, num_recv, src_tok_map};
  }

  /* ---------------------------------------------------------------- */
  /*  combine(expert_out, topk_weights) -> combined_x                  */
  /* ---------------------------------------------------------------- */
  torch::Tensor combine(const torch::Tensor& expert_out,
                        const torch::Tensor& topk_weights) {
    TORCH_CHECK(expert_out.is_cuda() && expert_out.dtype() == torch::kBFloat16);
    TORCH_CHECK(topk_weights.is_cuda() && topk_weights.dtype() == torch::kFloat32);

    stream = at::cuda::getCurrentCUDAStream().stream();

    index_t h_totalRecv = 0;
    HIP_CHECK(hipMemcpy(&h_totalRecv, args.totalRecvTokenNum,
                         sizeof(index_t), hipMemcpyDeviceToHost));

    args.inpTokenBuf = reinterpret_cast<hip_bfloat16*>(expert_out.data_ptr());
    args.weightsBuf  = reinterpret_cast<float*>(topk_weights.data_ptr());

    v1ll_rt::launch_combine(args, blockNum, warpPerBlock,
                            rdmaBlockNum, multiProcessorCount, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    int numTokens = args.curRankNumToken;
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(expert_out.device());
    auto combined = torch::from_blob(
        args.interNodeV1TokBufs.combineOut.localPtr,
        {cfg.MaxNumTokensToSendPerRank(), hiddenDim}, opts_bf16)
        .slice(0, 0, numTokens).clone();

    return combined;
  }
};

/* ================================================================== */
/*  Pybind11 module definition                                        */
/* ================================================================== */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Mori V1LL dispatch/combine kernel wrapper for rocSHMEM";

  m.def("shmem_get_uid", &v1ll_rt::shmem_get_uid,
        "Get a rocSHMEM unique ID for bootstrapping");
  m.def("shmem_init", &v1ll_rt::shmem_init,
        "Initialize rocSHMEM with a unique ID");
  m.def("shmem_barrier", &v1ll_rt::shmem_barrier,
        "Global rocSHMEM barrier");
  m.def("shmem_finalize", &v1ll_rt::shmem_finalize,
        "Finalize rocSHMEM");

  py::class_<V1LLBuffer>(m, "V1LLBuffer")
      .def(py::init<int, int, int, int, int, int, int, int, int>(),
           py::arg("rank"), py::arg("world_size"),
           py::arg("hidden_dim"), py::arg("num_expert_per_token"),
           py::arg("num_expert_per_rank"), py::arg("max_tokens_per_rank"),
           py::arg("warp_per_block") = 4, py::arg("block_num") = 32,
           py::arg("rdma_block_num") = 2)
      .def("dispatch", &V1LLBuffer::dispatch,
           py::arg("x"), py::arg("topk_idx"), py::arg("topk_weights"))
      .def("combine", &V1LLBuffer::combine,
           py::arg("expert_out"), py::arg("topk_weights"))
      .def("reset_counters", &V1LLBuffer::reset_counters);
}
