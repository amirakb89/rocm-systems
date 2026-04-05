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
#include <hip/hip_bfloat16.h>
using gpu_bfloat16_t = hip_bfloat16;
#include <random>
#include <vector>

enum class InitMode {
  Deterministic,
  Random
};

template <typename T>
class LLMoEData {
 public:

  // LL_DeepEP parameters
  const int num_tokens {0};
  const int hidden {0};
  const int num_topk {0};
  const int num_experts {0};

  // Input data (BF16 stored as T = short)
  T* X {nullptr};

  // Token to expert mapping
  int64_t *topk_idx {nullptr};

  // Routing weights per (token, expert) selection
  float *topk_weights {nullptr};

  // Number of tokens assigned to each expert
  std::vector<int> expert_token_count;

 private:
  InitMode init_mode {InitMode::Deterministic};

 public:
  LLMoEData(int num_tokens_, int hidden_, int num_topk_,
    int num_experts_, InitMode init_mode_ = InitMode::Deterministic)
      : num_tokens(num_tokens_), hidden(hidden_),
        num_topk(num_topk_), num_experts(num_experts_),
        expert_token_count(num_experts, 0), init_mode(init_mode_) {}

  ~LLMoEData() {
    if (X) CHECK_HIP(hipFree(X));
    if (topk_idx) CHECK_HIP(hipFree(topk_idx));
    if (topk_weights) CHECK_HIP(hipFree(topk_weights));
  }

  void generate_data() {

    size_t x_size_bytes = num_tokens * hidden * sizeof(T);
    size_t topk_idx_size_bytes = num_tokens * num_topk * sizeof(int64_t);
    size_t topk_weights_size_bytes = num_tokens * num_topk * sizeof(float);

    CHECK_HIP(hipMalloc(&X, x_size_bytes));
    CHECK_HIP(hipMalloc(&topk_idx, topk_idx_size_bytes));
    CHECK_HIP(hipMalloc(&topk_weights, topk_weights_size_bytes));

    int threads_per_block = 1024;
    int blocks_per_grid = num_tokens;

    int gpu_id {0};
    CHECK_HIP(hipGetDevice(&gpu_id));

    data_kernel<<<blocks_per_grid, threads_per_block>>>(X, hidden, gpu_id);
    CHECK_HIP(hipDeviceSynchronize());

    switch (init_mode) {
      case InitMode::Deterministic:
        generate_topk_deterministic();
        break;
      case InitMode::Random:
        generate_topk_random();
        break;
    }

    print();
  }

 private:
  void print() {
    std::cout << "LLMoEData:" << std::endl;
    std::cout << "  num_tokens: " << num_tokens << std::endl;
    std::cout << "  hidden: " << hidden << std::endl;
    std::cout << "  num_experts: " << num_experts << std::endl;
    std::cout << "  num_topk: " << num_topk << std::endl;
    std::cout << "  init_mode: "
              << (init_mode == InitMode::Deterministic
                      ? "Deterministic"
                      : "Random")
              << std::endl;
  }

  /**
   * GPU kernel to generate input data as valid BF16 values.
   * Each element = bf16(token_idx + 1.0f + rank * 0.001f + h * 0.0001f)
   * producing reasonable magnitudes for FP8 quantization.
   */
  __global__ static void data_kernel(T* X, int hidden, int rank) {
    int tkn_idx = blockIdx.x;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
      float val = static_cast<float>(tkn_idx + 1) +
                  rank * 0.001f + h * 0.0001f;
      gpu_bfloat16_t bf16_val(val);
      X[tkn_idx * hidden + h] = *reinterpret_cast<T*>(&bf16_val);
    }
  }

  void generate_topk_random() {
    size_t topk_idx_size = num_tokens * num_topk;
    std::vector<int64_t> h_topk_idx(topk_idx_size);
    std::vector<float> h_topk_weights(topk_idx_size);
    std::vector<int> expert_indices(num_experts);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> weight_dist(0.01f, 1.0f);

    for (int j = 0; j < num_experts; j++)
      expert_indices[j] = j;

    for (int i = 0; i < num_tokens; i++) {
      std::shuffle(expert_indices.begin(), expert_indices.end(), gen);
      float weight_sum = 0.0f;
      for (size_t k = 0; k < static_cast<size_t>(num_topk); k++) {
        h_topk_idx[i * num_topk + k] = expert_indices[k];
        h_topk_weights[i * num_topk + k] = weight_dist(gen);
        weight_sum += h_topk_weights[i * num_topk + k];
        expert_token_count[expert_indices[k]]++;
      }
      for (int k = 0; k < num_topk; k++)
        h_topk_weights[i * num_topk + k] /= weight_sum;
    }
    CHECK_HIP(hipMemcpy(topk_idx, h_topk_idx.data(),
                        topk_idx_size * sizeof(int64_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(topk_weights, h_topk_weights.data(),
                        topk_idx_size * sizeof(float), hipMemcpyHostToDevice));
  }

  void generate_topk_deterministic() {
    size_t topk_idx_size = num_tokens * num_topk;
    std::vector<int64_t> h_topk_idx(topk_idx_size);
    std::vector<float> h_topk_weights(topk_idx_size, 1.0f);

    for (int i = 0; i < num_tokens; i++) {
      for (int k = 0; k < num_topk; k++) {
        h_topk_idx[i * num_topk + k] = (i * num_topk + k) % num_experts;
        expert_token_count[h_topk_idx[i * num_topk + k]]++;
      }
    }
    CHECK_HIP(hipMemcpy(topk_idx, h_topk_idx.data(),
                        topk_idx_size * sizeof(int64_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(topk_weights, h_topk_weights.data(),
                        topk_idx_size * sizeof(float), hipMemcpyHostToDevice));
  }
};
