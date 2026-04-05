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

#include "../util.h"

/******************************************************************************
 * Buffer sizing matching DeepEP's LowLatencyLayout (config_hip.hpp).
 *
 * Dispatch message (FP8):
 *   int4 header (16B)  +  FP8 hidden (hidden B)  +  scales (hidden/128 * 4B)
 *
 * Combine message (BF16):
 *   int4 header (16B)  +  BF16 hidden (hidden * 2B)
 *
 * Send/recv buffers are sized for max(dispatch, combine) so they can be
 * reused between phases (dispatch runs first, then combine reuses the
 * same symmetric memory).  Double-buffered for pipelining.
 *****************************************************************************/

struct LLMoEBuffer {
  int num_sig_elems {0};

  void*    dispatch_send_buffer {nullptr};
  void*    dispatch_recv_buffer {nullptr};
  int64_t* dispatch_recv_count_buffer {nullptr};

  void*    combine_send_buffer {nullptr};
  void*    combine_recv_buffer {nullptr};
  int64_t* combine_recv_flag_buffer {nullptr};

  std::pair<int64_t*, int> clean_meta() {
    ASSERT(dispatch_recv_count_buffer == combine_recv_flag_buffer);
    return {dispatch_recv_count_buffer, num_sig_elems};
  }
};

template <typename T>
struct LLMoEBufferLayout {
  size_t total_bytes {0};
  LLMoEBuffer buffers[2];

  template <typename out_ptr_t = void*,
            typename count_ptr_t = uint8_t*,
            typename in_ptr_t = void*>
  out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
      return reinterpret_cast<out_ptr_t>(
          reinterpret_cast<count_ptr_t>(ptr) + count);
  }

  LLMoEBufferLayout(void* rdma_buffer, const int num_tokens, const int hidden,
      [[maybe_unused]] const int num_ranks, const int num_experts) {

    const int num_scales = hidden / 128;

    // Dispatch: int4 header + FP8 data + scales
    size_t num_bytes_per_dispatch_msg =
        sizeof(int4) + hidden + num_scales * sizeof(float);

    // Combine: int4 header + BF16 data
    size_t num_bytes_per_combine_msg =
        sizeof(int4) + hidden * sizeof(T);

    // Send buffers: dispatch sends per-token, combine sends per-expert-per-token
    size_t dispatch_send_buffer_bytes =
        num_tokens * num_bytes_per_dispatch_msg;
    size_t combine_send_buffer_bytes =
        num_experts * num_tokens * num_bytes_per_combine_msg;
    size_t send_buffer_bytes =
        std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);

    ASSERT(send_buffer_bytes % sizeof(int) == 0);
    total_bytes += send_buffer_bytes * 2;

    // Receive buffers: sized for all experts × tokens
    size_t dispatch_recv_buffer_bytes =
        num_experts * num_tokens * num_bytes_per_dispatch_msg;
    size_t combine_recv_buffer_bytes =
        num_experts * num_tokens * num_bytes_per_combine_msg;
    size_t recv_buffer_bytes =
        std::max(dispatch_recv_buffer_bytes, combine_recv_buffer_bytes);

    ASSERT(recv_buffer_bytes % sizeof(int) == 0);
    total_bytes += recv_buffer_bytes * 2;

    // Signaling buffers (dispatch counts / combine flags)
    size_t signaling_buffer_bytes = num_experts * sizeof(int64_t);
    total_bytes += signaling_buffer_bytes * 2;

    // Assign pointers (dispatch and combine share the same physical buffers)
    for (int i = 0; i < 2; ++ i) {
        buffers[i] = {
            num_experts,
            advance(rdma_buffer, send_buffer_bytes * i),
            advance(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * i),
            advance<int64_t*>(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * 2 +
                    signaling_buffer_bytes * i),
            advance(rdma_buffer, send_buffer_bytes * i),
            advance(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * i),
            advance<int64_t*>(rdma_buffer,
                    send_buffer_bytes * 2 + recv_buffer_bytes * 2 +
                    signaling_buffer_bytes * i)
        };
    }
  }
};

template <typename T>
size_t get_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden,
    int num_ranks, int num_experts) {
  LLMoEBufferLayout<T> ll_buffer_layout(nullptr, num_max_dispatch_tokens_per_rank,
                        hidden, num_ranks, num_experts);
  return ll_buffer_layout.total_bytes;
}
