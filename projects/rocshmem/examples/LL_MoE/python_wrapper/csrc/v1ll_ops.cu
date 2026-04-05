/*
 * Device code: includes V1LL kernel headers and provides host-callable
 * launch wrappers + rocSHMEM runtime helpers.
 *
 * Compiled by hipcc (via CUDAExtension with --variant rocm).
 */
#include <cstring>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <rocshmem/rocshmem.hpp>

#include "v1ll_api.hpp"

#include "LL_MoE_Dispatch_V1LL.hpp"
#include "LL_MoE_Combine_V1LL.hpp"

using namespace mori_v1ll;

namespace v1ll_rt {

/* ------------------------------------------------------------------ */
/*  rocSHMEM runtime wrappers                                         */
/* ------------------------------------------------------------------ */
void shmem_init(const std::vector<uint8_t>& root_uid_bytes,
                int rank, int num_ranks) {
  rocshmem_uniqueid_t uid;
  memcpy(&uid, root_uid_bytes.data(), sizeof(uid));
  rocshmem_init_attr_t attr;
  rocshmem_set_attr_uniqueid_args(rank, num_ranks, &uid, &attr);
  rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
  rocshmem_barrier_all();
}

std::vector<uint8_t> shmem_get_uid() {
  rocshmem_uniqueid_t uid;
  rocshmem_get_uniqueid(&uid);
  std::vector<uint8_t> out(sizeof(uid));
  memcpy(out.data(), &uid, sizeof(uid));
  return out;
}

void* shmem_alloc(size_t bytes) {
  return rocshmem_malloc(bytes);
}

void shmem_free_ptr(void* p) {
  if (p) rocshmem_free(p);
}

void shmem_barrier() {
  rocshmem_barrier_all();
}

void shmem_finalize() {
  rocshmem_finalize();
}

/* ------------------------------------------------------------------ */
/*  Kernel launchers                                                   */
/* ------------------------------------------------------------------ */
void launch_dispatch(EpDispatchCombineArgs<hip_bfloat16>& args,
                     int blockNum, int warpPerBlock,
                     int rdmaBlockNum, int mpCount,
                     hipStream_t stream) {
  LaunchDispatchV1LL<hip_bfloat16>(args, blockNum, warpPerBlock,
                                   rdmaBlockNum, mpCount, stream);
}

void launch_combine(EpDispatchCombineArgs<hip_bfloat16>& args,
                    int blockNum, int warpPerBlock,
                    int rdmaBlockNum, int mpCount,
                    hipStream_t stream) {
  LaunchCombineV1LL<hip_bfloat16>(args, blockNum, warpPerBlock,
                                  rdmaBlockNum, mpCount, stream);
}

}  // namespace v1ll_rt
