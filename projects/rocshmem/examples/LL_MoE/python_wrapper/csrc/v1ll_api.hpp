#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Forward-declare the args type so the binding can see it without hipcc
namespace mori_v1ll {
  template <typename T> struct EpDispatchCombineArgs;
  struct EpDispatchCombineConfig;
  struct SymmPtr;
  struct ShmemBufsInterNodeV1;
}

namespace v1ll_rt {

void shmem_init(const std::vector<uint8_t>& root_uid_bytes,
                int rank, int num_ranks);
std::vector<uint8_t> shmem_get_uid();
void* shmem_alloc(size_t bytes);
void shmem_free_ptr(void* p);
void shmem_barrier();
void shmem_finalize();

void launch_dispatch(mori_v1ll::EpDispatchCombineArgs<hip_bfloat16>& args,
                     int blockNum, int warpPerBlock,
                     int rdmaBlockNum, int mpCount,
                     hipStream_t stream);

void launch_combine(mori_v1ll::EpDispatchCombineArgs<hip_bfloat16>& args,
                    int blockNum, int warpPerBlock,
                    int rdmaBlockNum, int mpCount,
                    hipStream_t stream);

}  // namespace v1ll_rt
