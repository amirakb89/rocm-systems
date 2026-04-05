/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include <mpi.h>
#include <cmath>

#include "LL_MoE_Data.hpp"
#include "LL_MoE_Buffers.hpp"
#include "LL_MoE_Kernels.hpp"

using namespace rocshmem;

template<typename T>
class LLMoE {
 private:
  int rank {0}, num_ranks {0};
  int device_id {0};

  int64_t num_rdma_bytes {0};
  void*   rdma_buffer_ptr {nullptr};

  void* workspace {nullptr};

  const int num_tokens {0};
  const int hidden {0};
  const int num_topk {0};
  const int num_experts {0};

  LLMoEData<T> ll_moe_data;

  int ll_buffer_idx {0};

  // Dispatch output: raw FP8 + scales (dequanted to BF16 by dequant kernel)
  uint8_t*  packed_recv_fp8 {nullptr};
  float*    packed_recv_scales {nullptr};
  T*        packed_recv_x {nullptr};
  int*      packed_recv_src_info {nullptr};
  int64_t*  packed_recv_layout_range {nullptr};
  int*      packed_recv_count {nullptr};
  int*      global_atomic_counter {nullptr};

  T*        combined_x {nullptr};

  hipStream_t stream;

  InitMode init_mode {InitMode::Deterministic};

 public:
  LLMoE(int num_tokens_, int hidden_, int num_topk_, int num_experts_,
    InitMode init_mode_ = InitMode::Deterministic)
      : num_tokens(num_tokens_), hidden(hidden_), num_topk(num_topk_),
        num_experts(num_experts_),
        ll_moe_data(num_tokens_, hidden_, num_topk_, num_experts_,init_mode_),
        init_mode(init_mode_) {

    comm_init();
    std::cout << "Rank " << rank << " using GPU " << device_id << std::endl;
    CHECK_HIP(hipStreamCreate(&stream));

    ASSERT(num_experts % num_ranks == 0);
    ASSERT(hidden % 128 == 0);

    ll_moe_data.generate_data();

    CHECK_HIP(hipExtMallocWithFlags(&workspace, NUM_WORKSPACE_BYTES,
              hipDeviceMallocUncached));
    CHECK_HIP(hipMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, stream));

    num_rdma_bytes = get_rdma_size_hint<T>(
        num_tokens, hidden, num_ranks, num_experts);
    rdma_buffer_ptr = rocshmem_malloc(num_rdma_bytes);
    if (rdma_buffer_ptr == nullptr) {
      std::cerr << "Rank " << rank
                << ": Error in rocshmem_malloc (" << num_rdma_bytes
                << " bytes). Aborting." << std::endl;
      comm_finalize();
      exit(EXIT_FAILURE);
    }

    CHECK_HIP(hipMemsetAsync(rdma_buffer_ptr, 0, num_rdma_bytes, stream));

    allocate_dispatch_buffers();

    CHECK_HIP(hipStreamSynchronize(stream));
  }

  ~LLMoE() {
    CHECK_HIP(hipFree(workspace));
    CHECK_HIP(hipFree(packed_recv_fp8));
    CHECK_HIP(hipFree(packed_recv_scales));
    CHECK_HIP(hipFree(packed_recv_x));
    CHECK_HIP(hipFree(packed_recv_src_info));
    CHECK_HIP(hipFree(packed_recv_layout_range));
    CHECK_HIP(hipFree(packed_recv_count));
    CHECK_HIP(hipFree(combined_x));
    CHECK_HIP(hipFree(global_atomic_counter));
    CHECK_HIP(hipStreamDestroy(stream));
    rocshmem_free(rdma_buffer_ptr);
    comm_finalize();
  }

  int get_rank() const { return rank; }
  int get_num_ranks() const { return num_ranks; }

  void ll_dispatch() {
    LLMoEBufferLayout<T> ll_layout(rdma_buffer_ptr, num_tokens,
                       hidden, num_ranks, num_experts);
    LLMoEBuffer& buffer      = ll_layout.buffers[ll_buffer_idx];
    LLMoEBuffer& next_buffer = ll_layout.buffers[ll_buffer_idx ^= 1];

    ll_kernels::dispatch<T>(packed_recv_fp8, packed_recv_scales,
      packed_recv_src_info,
      packed_recv_layout_range, packed_recv_count, global_atomic_counter,
      buffer.dispatch_recv_buffer, buffer.dispatch_recv_count_buffer,
      buffer.dispatch_send_buffer, ll_moe_data.X, ll_moe_data.topk_idx,
      next_buffer.clean_meta().first, next_buffer.clean_meta().second,
      num_tokens, hidden, num_topk, num_experts, rank, num_ranks, workspace,
      stream);
  }

  void ll_dequant() {
    int num_local_experts = num_experts / num_ranks;
    int stride_tokens = num_ranks * num_tokens;
    ll_kernels::dequant_fp8_to_bf16<T>(
        packed_recv_x, packed_recv_fp8, packed_recv_scales,
        packed_recv_count, num_local_experts, stride_tokens,
        hidden, stream);
  }

  void ll_combine() {
    LLMoEBufferLayout<T> ll_layout(rdma_buffer_ptr, num_tokens,
                       hidden, num_ranks, num_experts);
    LLMoEBuffer& buffer      = ll_layout.buffers[ll_buffer_idx];
    LLMoEBuffer& next_buffer = ll_layout.buffers[ll_buffer_idx ^= 1];

    ll_kernels::combine<T>(combined_x, buffer.combine_recv_buffer,
      buffer.combine_recv_flag_buffer, buffer.combine_send_buffer,
      packed_recv_x, ll_moe_data.topk_idx, ll_moe_data.topk_weights,
      packed_recv_src_info, packed_recv_layout_range,
      global_atomic_counter,
      next_buffer.clean_meta().first, next_buffer.clean_meta().second,
      num_tokens, num_topk, hidden, num_experts, rank, num_ranks,
      workspace, stream);

    if (init_mode == InitMode::Deterministic) {
      CHECK_HIP(hipStreamSynchronize(stream));
      verify_combined_x(combined_x);
    }
  }

 private:
  void comm_init() {
    int mpi_rank {0}, mpi_size {0};
    int ret {0};
    int provided {0};
    MPI_Init_thread (nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
      std::cerr << "MPI_THREAD_MULTIPLE support disabled.\n";
    MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

    int device_count {0};
    CHECK_HIP(hipGetDeviceCount(&device_count));
    CHECK_HIP(hipSetDevice(mpi_rank % device_count));
    CHECK_HIP(hipGetDevice(&device_id));

    rocshmem_uniqueid_t uid;
    rocshmem_init_attr_t attr;
    if (mpi_rank == 0) {
      ret = rocshmem_get_uniqueid (&uid);
      if (ret != ROCSHMEM_SUCCESS) {
        std::cout << mpi_rank << ": Error in rocshmem_get_uniqueid. Aborting." << std::endl;
        MPI_Abort (MPI_COMM_WORLD, ret);
      }
    }
    MPI_Bcast (&uid, sizeof(rocshmem_uniqueid_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    ret = rocshmem_set_attr_uniqueid_args(mpi_rank, mpi_size, &uid, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << mpi_rank << ": Error in rocshmem_set_attr_uniqueid_args. Aborting" << std::endl;
      MPI_Abort (MPI_COMM_WORLD, ret);
    }
    ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << mpi_rank << ": Error in rocshmem_init_attr. Aborting." << std::endl;
      MPI_Abort (MPI_COMM_WORLD, ret);
    }
    rank      = rocshmem_my_pe();
    num_ranks = rocshmem_n_pes();
  }

  void comm_finalize() {
    rocshmem_finalize();
    MPI_Finalize();
  }

  void allocate_dispatch_buffers() {
    int num_local_experts = num_experts / num_ranks;
    int stride_tokens = num_ranks * num_tokens;
    int num_scales = hidden / 128;

    size_t packed_fp8_bytes = (size_t)num_local_experts * stride_tokens * hidden;
    size_t packed_scales_bytes = (size_t)num_local_experts * stride_tokens * num_scales * sizeof(float);
    size_t packed_recv_x_bytes = (size_t)num_local_experts * stride_tokens * hidden * sizeof(T);
    size_t packed_recv_src_info_bytes = (size_t)num_local_experts * stride_tokens * sizeof(int);
    size_t packed_recv_layout_range_bytes = (size_t)num_local_experts * num_ranks * sizeof(int64_t);
    size_t packed_recv_count_bytes = (size_t)num_local_experts * sizeof(int);
    size_t combined_x_bytes = (size_t)num_tokens * hidden * sizeof(T);

    CHECK_HIP(hipMalloc(&packed_recv_fp8, packed_fp8_bytes));
    CHECK_HIP(hipMalloc(&packed_recv_scales, packed_scales_bytes));
    CHECK_HIP(hipMalloc(&packed_recv_x, packed_recv_x_bytes));
    CHECK_HIP(hipMalloc(&packed_recv_src_info, packed_recv_src_info_bytes));
    CHECK_HIP(hipMalloc(&packed_recv_layout_range, packed_recv_layout_range_bytes));
    CHECK_HIP(hipMalloc(&packed_recv_count, packed_recv_count_bytes));
    CHECK_HIP(hipMalloc(&combined_x, combined_x_bytes));
    CHECK_HIP(hipMalloc(&global_atomic_counter, sizeof(int)));
    CHECK_HIP(hipMemsetAsync(global_atomic_counter, 0, sizeof(int), stream));
  }

  void verify_combined_x(void* combined_x_ptr) {
    size_t total = static_cast<size_t>(num_tokens) * hidden;
    std::vector<T> h_combined(total);
    std::vector<T> h_X(total);
    CHECK_HIP(hipMemcpy(h_combined.data(), combined_x_ptr,
              total * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_X.data(), ll_moe_data.X,
              total * sizeof(T), hipMemcpyDeviceToHost));

    int mismatches = 0;
    constexpr float kRelTol = 0.05f;
    for (int i = 0; i < num_tokens; i++) {
      for (int h = 0; h < hidden; h++) {
        gpu_bfloat16_t expected_bf16 =
            *reinterpret_cast<gpu_bfloat16_t*>(&h_X[i * hidden + h]);
        float expected = static_cast<float>(expected_bf16) * num_topk;
        gpu_bfloat16_t actual_bf16 =
            *reinterpret_cast<gpu_bfloat16_t*>(&h_combined[i * hidden + h]);
        float actual = static_cast<float>(actual_bf16);
        float abs_err = fabsf(actual - expected);
        float rel_err = (fabsf(expected) > 1e-6f) ?
                        abs_err / fabsf(expected) : abs_err;
        if (rel_err > kRelTol && abs_err > 1e-3f) {
          if (mismatches < 5)
            std::cout << "[rank " << rank << "] Mismatch at Token " << i
                      << ", Hidden " << h << ": Expected " << expected
                      << ", Actual " << actual << ", RelErr " << rel_err << std::endl;
          mismatches++;
        }
      }
    }
    if (mismatches > 0)
      std::cout << "[rank " << rank << "] Verification: "
                << mismatches << " / " << total
                << " elements exceed " << kRelTol * 100 << "% tolerance" << std::endl;
  }
};
