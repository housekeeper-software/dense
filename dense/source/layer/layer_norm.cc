#include "layer/layer_norm.h"
#include "layer/init.h"
#include "math/vec_math.h"
#include <iostream>
#include <numeric>

namespace dense {
namespace {
std::pair<int64_t, int64_t>
_check_nested_layer_norm_inputs(const dense::Tensor &input,
                                const std::vector<int64_t> &normalized_shape) {
  const size_t normalized_ndim = normalized_shape.size();
  int64_t N = 1;
  for (const auto &i : normalized_shape) {
    N *= i;
  }
  const int64_t M = input.numel() / N;
  return std::make_pair(M, N);
}
} // namespace

LayerNorm::LayerNorm(Context *ctx, const std::string &name,
                     const std::vector<int64_t> &normalized_shape,
                     float epsilon, bool elementwise_affine, bool has_bias)
    : Layer(ctx, name), normalized_shape_(normalized_shape), epsilon_(epsilon),
      elementwise_affine_(elementwise_affine), has_bias_(has_bias) {

  if (elementwise_affine_) {
    // 这个层有可学习参数,gamma ,beta
    RegisterParam();
  }

  if (elementwise_affine_) {
    // gamma 通常初始化为 1
    W_ = Tensor::empty(DType::kFloat32, normalized_shape_);
    init::ones_(W_);

    if (has_bias_) {
      // beta 通常初始化为 0
      b_ = Tensor::zeros(DType::kFloat32, normalized_shape_);
    }
  }
}

dense::Tensor LayerNorm::forward(const dense::Tensor &input) {
  if (input.sizes().size() < normalized_shape_.size()) {
    throw std::runtime_error("输入维度小于归一化维度.");
  }

  if (ctx()->device.is_blas()) {
    return forward_blas(input);
  }
  return forward_cpu(input);
}

dense::Tensor LayerNorm::backward(const dense::Tensor &grad_output) {
  if (ctx()->device.is_blas()) {
    return backward_blas(grad_output);
  }
  return backward_cpu(grad_output);
}

dense::Tensor LayerNorm::forward_cpu(const dense::Tensor &input) {
  auto output = dense::Tensor::zeros_like(input);

  // 我们要对维度进行缩并，最终形成 [M, N], N：归一化维度
  auto [M, N] = _check_nested_layer_norm_inputs(input, normalized_shape_);

  mean_ = dense::Tensor::zeros(input.dtype(), {M});
  rstd_ = dense::Tensor::zeros(input.dtype(), {M});
  x_norm_ = dense::Tensor::zeros_like(input);

  auto w_ptr = W_.const_data_as<float>();
  auto b_ptr = b_.const_data_as<float>();

  auto mean_ptr = mean_.mutable_data_as<float>();
  auto rstd_ptr = rstd_.mutable_data_as<float>();
  auto in_ptr = input.const_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();
  auto x_norm_ptr = x_norm_.mutable_data_as<float>();

  for (int64_t m = 0; m < M; ++m) {
    auto in_bt = in_ptr + m * N;
    auto out_bt = out_ptr + m * N;
    auto x_norm_bt = x_norm_ptr + m * N;

    float mean = 0.0f;
    for (int64_t i = 0; i < N; ++i) {
      mean += in_bt[i];
    }
    mean_ptr[m] = mean / N; // 均值

    float var = 0.0f;
    for (int64_t i = 0; i < N; ++i) {
      auto x_shift = in_bt[i] - mean_ptr[m];
      var += x_shift * x_shift;
    }
    var = var / N; // 方差

    rstd_ptr[m] = 1.0f / std::sqrt(var + epsilon_);

    for (int64_t i = 0; i < N; ++i) {
      // x_hat = (x-mean)/sqrt(var+eplison)
      x_norm_bt[i] = (in_bt[i] - mean_ptr[m]) * rstd_ptr[m];
    }

    if (elementwise_affine_) {
      // y = x_hat*gamma + beta
      for (int64_t i = 0; i < N; ++i) {
        // 归一化结果
        out_bt[i] = x_norm_bt[i] * w_ptr[i];
        if (has_bias_) {
          // 如果有偏置
          out_bt[i] += b_ptr[i];
        }
      }
    } else {
      // 不需要仿射变换，我们直接将归一化结果输出
      std::copy_n(x_norm_bt, N, out_bt);
    }
  }
  return output;
}

dense::Tensor LayerNorm::forward_blas(const dense::Tensor &input) {
  auto output = dense::Tensor::zeros_like(input);

  // 我们要对维度进行缩并，最终形成 [M, N], N：归一化维度
  auto [M, N] = _check_nested_layer_norm_inputs(input, normalized_shape_);

  mean_ = dense::Tensor::zeros(input.dtype(), {M});
  rstd_ = dense::Tensor::zeros(input.dtype(), {M});
  x_norm_ = dense::Tensor::zeros_like(input);

  auto w_ptr = W_.const_data_as<float>();
  auto b_ptr = b_.const_data_as<float>();

  auto mean_ptr = mean_.mutable_data_as<float>();
  auto rstd_ptr = rstd_.mutable_data_as<float>();
  auto in_ptr = input.const_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();
  auto x_norm_ptr = x_norm_.mutable_data_as<float>();

  std::vector<float> ones(std::max(M, N), 1.0f);

  // 计算均值
  // mean = sum(x) / N
  vec::sgemv_blas(M, N, 1.0f, in_ptr, N, ones.data(), 1, 0.0f, mean_ptr, 1);
  vec::sscal_blas(M, 1.0f / N, mean_ptr, 1);

  for (int64_t m = 0; m < M; ++m) {
    auto in_bt = input.const_data_as<float>() + m * N;
    auto out_bt = output.mutable_data_as<float>() + m * N;
    auto x_norm_bt = x_norm_ptr + m * N;

    // 1. x_hat = x
    vec::scopy_blas(N, in_bt, 1, x_norm_bt, 1);

    // 2. y = x - mean
    // 这里我们直接在 x_hat 上进行操作
    vec::saxpy_blas(N, -mean_ptr[m], ones.data(), 1, x_norm_bt, 1);

    // 3. 计算方差
    // var = sum((x-mean)^2) / N
    // sum(y-mean)^2, 我们用自己和自己内积来实现
    float var = vec::sdot_blas(N, x_norm_bt, 1, x_norm_bt, 1);
    var = var / N; // 方差

    // 4. 计算 rstd = 1/sqrt(var+epsilon)
    rstd_ptr[m] = 1.0f / std::sqrt(var + epsilon_);

    // 5. x_hat = (x-mean) / sqrt(var+epsilon)
    // 这里我们直接在 x_hat 上进行操作
    vec::sscal_blas(N, rstd_ptr[m], x_norm_bt, 1);

    // 6. y = x_hat * gamma + beta
    // 如果有仿射变换
    if (elementwise_affine_) {
      // output = x_hat * gamma
      // 这里我们直接在 output 上进行操作
      vec::shdm_blas(N, w_ptr, x_norm_bt, out_bt);
      if (has_bias_) {
        // output = output + beta
        vec::saxpy_blas(N, 1.0f, b_ptr, 1, out_bt, 1);
      }
    } else {
      // 不需要仿射变换，我们直接将归一化结果输出
      vec::scopy_blas(N, x_norm_bt, 1, out_bt, 1);
    }
  }
  return output;
}

dense::Tensor LayerNorm::backward_cpu(const dense::Tensor &grad_output) {
  if (elementwise_affine_) {
    if (!grad_W_.is_defined()) {
      grad_W_ = dense::Tensor::zeros_like(W_);
    }
    if (has_bias_ && !grad_b_.is_defined()) {
      grad_b_ = dense::Tensor::zeros_like(b_);
    }
  }

  auto x_hat = x_norm_; // 前向传播时的 x_norm
  auto grad_input = dense::Tensor::zeros_like(x_hat);

  auto [M, N] = _check_nested_layer_norm_inputs(grad_output, normalized_shape_);

  auto grad_w_ptr = grad_W_.mutable_data_as<float>();
  auto w_ptr = W_.const_data_as<float>();
  auto grad_b_ptr = grad_b_.mutable_data_as<float>();

  auto mean_ptr = mean_.const_data_as<float>();
  auto rstd_ptr = rstd_.const_data_as<float>();
  auto grad_out_ptr = grad_output.const_data_as<float>();
  auto grad_in_ptr = grad_input.mutable_data_as<float>();
  auto x_hat_ptr = x_hat.const_data_as<float>();

  std::vector<float> dl_dx_hat(N, 0.0f);

  for (int64_t m = 0; m < M; ++m) {
    // 这三个形状相同
    auto x_hat_bt = x_hat_ptr + m * N;
    auto grad_out_bt = grad_out_ptr + m * N;
    auto grad_in_bt = grad_in_ptr + m * N;

    auto mean_bt = mean_ptr[m];
    auto rstd_bt = rstd_ptr[m];

    float dl_dx_hat_sum = 0.0f;
    float dl_dx_hat_dot_x_hat = 0.0f;

    for (int64_t i = 0; i < N; ++i) {
      // dL/dx_hat = dL/dy * gamma
      if (elementwise_affine_) {
        dl_dx_hat[i] = grad_out_bt[i] * w_ptr[i];
      } else {
        dl_dx_hat[i] = grad_out_bt[i]; // 如果没有 gamma
      }
      // Σ(dL/dx_hat),最后计算的时候要用到的中间值
      dl_dx_hat_sum += dl_dx_hat[i];
      // Σ(x_hat * dL/dx_hat)
      dl_dx_hat_dot_x_hat += x_hat_bt[i] * dl_dx_hat[i];

      if (elementwise_affine_) {
        // dL/dW = Σ(dL/dy * x_hat)
        grad_w_ptr[i] += grad_out_bt[i] * x_hat_bt[i];
        if (has_bias_) {
          // dL/db = Σ(dL/dy)
          grad_b_ptr[i] += grad_out_bt[i];
        }
      }
    }

    float rtsd_mean = (1.0 / N) * rstd_bt;

    for (int64_t i = 0; i < N; ++i) {
      grad_in_bt[i] = rtsd_mean * (N * dl_dx_hat[i] - dl_dx_hat_sum -
                                   x_hat_bt[i] * dl_dx_hat_dot_x_hat);
    }
  }
  return grad_input;
}

dense::Tensor LayerNorm::backward_blas(const dense::Tensor &grad_output) {
  if (elementwise_affine_) {
    // 如果有仿射变换，确保梯度张量已定义
    if (!grad_W_.is_defined()) {
      grad_W_ = dense::Tensor::zeros_like(W_);
    }
    if (has_bias_ && !grad_b_.is_defined()) {
      grad_b_ = dense::Tensor::zeros_like(b_);
    }
  }

  auto x_hat = x_norm_; // 前向传播时的 x_norm
  auto grad_input = dense::Tensor::zeros_like(x_hat);

  auto [M, N] = _check_nested_layer_norm_inputs(grad_output, normalized_shape_);

  auto grad_w_ptr = grad_W_.mutable_data_as<float>();
  auto w_ptr = W_.const_data_as<float>();
  auto grad_b_ptr = grad_b_.mutable_data_as<float>();

  auto mean_ptr = mean_.const_data_as<float>();
  auto rstd_ptr = rstd_.const_data_as<float>();
  auto grad_out_ptr = grad_output.const_data_as<float>();
  auto grad_in_ptr = grad_input.mutable_data_as<float>();
  auto x_hat_ptr = x_hat.const_data_as<float>();

  std::vector<float> ones(std::max(M, N), 1.0f);

  if (elementwise_affine_) {
    // dL/dW = Σ(dL/dy * x_hat)
    // 减少内存分配
    if (hdm_.sizes() != x_hat.sizes()) {
      hdm_ = dense::Tensor::zeros_like(x_hat);
    }
    // 先计算哈达玛积
    vec::shdm_blas(M * N, grad_output.const_data_as<float>(), x_hat_ptr,
                   hdm_.mutable_data_as<float>());

    // 通过矩阵与全 1 向量的乘法来实现求和，这里要转置，因为我们按照列求和
    // beta = 1.0f，这个很重要，因为我们要对梯度进行累加
    // 以便满足训练时的梯度累加需求
    vec::sgemv_transpose_blas(M, N, 1.0f, hdm_.const_data_as<float>(), N,
                              ones.data(), 1, 1.0f, grad_w_ptr, 1);
    if (has_bias_) {
      // dL/db = Σ(dL/dy)
      // beta = 1.0f，这个很重要，因为我们要对梯度进行累加
      // 以便满足训练时的梯度累加需求
      vec::sgemv_transpose_blas(M, N, 1.0f, grad_output.const_data_as<float>(),
                                N, ones.data(), 1, 1.0f, grad_b_ptr, 1);
    }
  }

  std::vector<float> dl_dx_hat(N, 0.0f);

  for (int64_t m = 0; m < M; ++m) {
    // 这三个形状相同
    auto x_hat_bt = x_hat_ptr + m * N;
    auto grad_out_bt = grad_out_ptr + m * N;
    auto grad_in_bt = grad_in_ptr + m * N;

    auto mean_bt = mean_ptr[m];
    auto rstd_bt = rstd_ptr[m];

    float dl_dx_hat_sum = 0.0f;
    float dl_dx_hat_dot_x_hat = 0.0f;

    // 计算 dL/dx_hat
    if (elementwise_affine_) {
      vec::shdm_blas(N, grad_out_bt, w_ptr, dl_dx_hat.data());
    } else {
      vec::scopy_blas(N, grad_out_bt, 1, dl_dx_hat.data(), 1);
    }
    // 计算 Σ(dL/dx_hat)
    dl_dx_hat_sum = vec::sdot_blas(N, dl_dx_hat.data(), 1, ones.data(), 1);
    // 计算 Σ(x_hat * dL/dx_hat)
    dl_dx_hat_dot_x_hat = vec::sdot_blas(N, x_hat_bt, 1, dl_dx_hat.data(), 1);

    float rtsd_mean = (1.0 / N) * rstd_bt;

    for (int64_t i = 0; i < N; ++i) {
      grad_in_bt[i] = rtsd_mean * (N * dl_dx_hat[i] - dl_dx_hat_sum -
                                   x_hat_bt[i] * dl_dx_hat_dot_x_hat);
    }
  }
  return grad_input;
}
} // namespace dense