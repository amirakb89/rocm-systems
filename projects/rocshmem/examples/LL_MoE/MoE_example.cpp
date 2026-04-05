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

#include "LL_MoE.hpp"
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdio>

using dtype = short;

/******************************************************************************
 * This example mirrors the communication algorithm of DeepEP's low-latency
 * (LL) internode dispatch/combine kernels (internode_ll.hip):
 *
 *   Dispatch:  BF16 input → FP8 E4M3_FNUZ quantization (per-128-channel
 *              scaling) → RDMA put → receive + dequant to BF16
 *
 *   Combine:   BF16 expert output → RDMA put → weighted reduction
 *              (float accumulate with topk_weights → BF16 output)
 *
 * Key algorithmic elements ported from DeepEP:
 *   - FP8 quantization with quarter-warp amax reduce (dispatch)
 *   - int4 (16-byte) message headers
 *   - 2 warp-groups × 8 warps thread configuration (ROCm)
 *   - Weighted combine with topk_weights
 *****************************************************************************/

void print_usage(const char* prog_name)
{
  std::cerr
        << "Usage: " << prog_name << " [options]\n"
        << "Options:\n"
        << "  -u                 Print this usage message and exit\n"
        << "  -n <num_tokens>    Number of tokens (default: 128)\n"
        << "  -h <hidden>        Hidden size (default: 7168, must be multiple of 128)\n"
        << "  -k <num_topk>      Top-k value (default: 8)\n"
        << "  -e <num_experts>   Number of experts (default: 288)\n"
        << "  -i <iterations>    Number of iterations (default: 10)\n"
        << "  -m <r|d>           Buffer Init mode:\n"
        << "                     r = random (default)\n"
        << "                     d = deterministic\n";
}

int main (int argc, char **argv)
{
  int rank, num_ranks;

  int num_tokens  = 128;
  int hidden      = 7168;
  int num_topk    = 8;
  int num_experts = 288;

  int num_iterations = 10;

  InitMode init_mode = InitMode::Random;

  int opt;
  while ((opt = getopt(argc, argv, "n:h:k:e:i:m:u")) != -1) {
    switch (opt) {
      case 'n':
        num_tokens = atoi(optarg);
        break;
      case 'h':
        hidden = atoi(optarg);
        break;
      case 'k':
        num_topk = atoi(optarg);
        break;
      case 'e':
        num_experts = atoi(optarg);
        break;
      case 'i':
        num_iterations = atoi(optarg);
        break;
      case 'u':
        print_usage(argv[0]);
        return EXIT_SUCCESS;
      case 'm':
        if (optarg[0] == 'r' && optarg[1] == '\0') {
          init_mode = InitMode::Random;
        } else if (optarg[0] == 'd' && optarg[1] == '\0') {
          init_mode = InitMode::Deterministic;
        } else {
          std::cerr << "Invalid value for -m: " << optarg << std::endl;
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        break;
      case '?':
        if (optopt == 'n' || optopt == 'h' || optopt == 'k' ||
            optopt == 'e' || optopt == 'i' || optopt == 'm') {
          std::cerr << "Option -" << static_cast<char>(optopt)
                    << " requires an argument." << std::endl;
        } else {
          std::cerr << "Unknown option -"
                  << static_cast<char>(optopt) << std::endl;
        }
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
  }

  if (hidden % 128 != 0) {
    std::cerr << "hidden (" << hidden << ") must be a multiple of 128" << std::endl;
    return EXIT_FAILURE;
  }

  LLMoE<dtype> ll_moe(num_tokens, hidden, num_topk, num_experts, init_mode);

  rank = ll_moe.get_rank();
  num_ranks = ll_moe.get_num_ranks();

  std::cout << "rank: " << rank << ", n_RANKs: " << num_ranks << std::endl;

  constexpr int warmup_iters = 3;
  int bench_iters = std::max(1, num_iterations - warmup_iters);

  for (int iter = 0; iter < warmup_iters && iter < num_iterations; iter++) {
      ll_moe.ll_dispatch();
      ll_moe.ll_dequant();
      ll_moe.ll_combine();
  }

  std::vector<float> dispatch_ms(bench_iters);
  std::vector<float> dequant_ms(bench_iters);
  std::vector<float> combine_ms(bench_iters);
  std::vector<float> e2e_ms(bench_iters);

  for (int iter = 0; iter < bench_iters; iter++) {
      hipEvent_t d_start, d_stop, q_start, q_stop, c_start, c_stop;
      (void)hipEventCreate(&d_start);
      (void)hipEventCreate(&d_stop);
      (void)hipEventCreate(&q_start);
      (void)hipEventCreate(&q_stop);
      (void)hipEventCreate(&c_start);
      (void)hipEventCreate(&c_stop);

      (void)hipEventRecord(d_start);
      ll_moe.ll_dispatch();
      (void)hipEventRecord(d_stop);

      (void)hipEventRecord(q_start);
      ll_moe.ll_dequant();
      (void)hipEventRecord(q_stop);

      (void)hipEventRecord(c_start);
      ll_moe.ll_combine();
      (void)hipEventRecord(c_stop);

      (void)hipEventSynchronize(c_stop);

      (void)hipEventElapsedTime(&dispatch_ms[iter], d_start, d_stop);
      (void)hipEventElapsedTime(&dequant_ms[iter], q_start, q_stop);
      (void)hipEventElapsedTime(&combine_ms[iter], c_start, c_stop);
      e2e_ms[iter] = dispatch_ms[iter] + dequant_ms[iter] + combine_ms[iter];

      (void)hipEventDestroy(d_start);
      (void)hipEventDestroy(d_stop);
      (void)hipEventDestroy(q_start);
      (void)hipEventDestroy(q_stop);
      (void)hipEventDestroy(c_start);
      (void)hipEventDestroy(c_stop);
  }

  /*
   * BW computation matching DeepEP's byte counting:
   *
   * Dispatch (FP8):  each selection transfers
   *   sizeof(int4) + hidden + (hidden/128)*sizeof(float)  bytes
   *
   * Combine (BF16):  each selection transfers
   *   hidden * sizeof(dtype)  bytes  (header is padding, not transferred)
   */
  const int num_scales = hidden / 128;
  const size_t disp_msg_bytes = sizeof(int4) + hidden +
                                num_scales * sizeof(float);
  const size_t comb_msg_bytes = static_cast<size_t>(hidden) * sizeof(dtype);

  size_t total_selections = static_cast<size_t>(num_tokens) * num_topk;
  double dispatch_bytes = static_cast<double>(total_selections * disp_msg_bytes);
  double combine_bytes  = static_cast<double>(total_selections * comb_msg_bytes);

  auto avg = [](const std::vector<float>& v) {
      return std::accumulate(v.begin(), v.end(), 0.0f) /
             static_cast<float>(v.size());
  };
  auto minv = [](const std::vector<float>& v) {
      return *std::min_element(v.begin(), v.end());
  };
  auto maxv = [](const std::vector<float>& v) {
      return *std::max_element(v.begin(), v.end());
  };

  float d_avg = avg(dispatch_ms);
  float c_avg = avg(combine_ms);
  float e_avg = avg(e2e_ms);
  double d_bw = dispatch_bytes / 1e9 / (d_avg / 1e3);
  double c_bw = combine_bytes  / 1e9 / (c_avg / 1e3);

  float q_avg = avg(dequant_ms);

  printf("[rank %d] Dispatch (FP8):  avg=%.1f us, min=%.1f us, max=%.1f us, "
         "BW=%.2f GB/s (%zu B/selection)\n",
         rank, d_avg * 1e3, minv(dispatch_ms) * 1e3, maxv(dispatch_ms) * 1e3,
         d_bw, disp_msg_bytes);
  printf("[rank %d] Dequant:         avg=%.1f us, min=%.1f us, max=%.1f us\n",
         rank, q_avg * 1e3, minv(dequant_ms) * 1e3, maxv(dequant_ms) * 1e3);
  printf("[rank %d] Combine  (BF16): avg=%.1f us, min=%.1f us, max=%.1f us, "
         "BW=%.2f GB/s (%zu B/selection)\n",
         rank, c_avg * 1e3, minv(combine_ms) * 1e3, maxv(combine_ms) * 1e3,
         c_bw, comb_msg_bytes);
  printf("[rank %d] E2E:             avg=%.1f us, min=%.1f us, max=%.1f us, "
         "BW=%.2f GB/s\n",
         rank, e_avg * 1e3, minv(e2e_ms) * 1e3, maxv(e2e_ms) * 1e3,
         (dispatch_bytes + combine_bytes) / 1e9 / (e_avg / 1e3));
  fflush(stdout);

  if (rank == 0) {
      std::cout << "\nAll " << num_iterations
                << " iterations completed successfully!" << std::endl;
  }

  return 0;
}
