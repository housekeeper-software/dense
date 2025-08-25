#include "layer/linear.h"
#include "layer/init.h"
#include "math/vec_math.h"
#include <numeric>
#include <stdexcept>

namespace dense {

Linear::Linear(Context *ctx, const std::string &name, int64_t in_features,
               int64_t out_features, bool has_bias)
    : Layer(ctx, name), in_features_(in_features), out_features_(out_features),
      has_bias_(has_bias) {
  // 线性层有可学习参数 W_,b_(if has_bias_=true)
  RegisterParam();
  // torch::nn::Linear 的权重形状是 [out_features, in_features]
  // 权重和偏置初始化参考 pytorch nn.Linear
  W_ = Tensor::empty(DType::kFloat32, {out_features_, in_features_});
  init::kaiming_normal_(W_, std::sqrt(5), init::FanModeType::kFanIn,
                        init::NonlinearityType::kLeakyReLU);
  if (has_bias) {
    b_ = Tensor::empty(DType::kFloat32, {out_features_});

    auto fan = init::_calculate_fan_in_and_fan_out(W_);
    auto fan_in = std::get<0>(fan);
    auto bound = fan_in > 0 ? 1.0 / std::sqrt(fan_in) : 0;
    init::uniform_(b_, -bound, bound);
  }
}

Tensor Linear::forward(const Tensor &input) {
  if (input.dim() < 2) {
    throw std::runtime_error("The expected input should be at least 2D.");
  }
  if (is_training()) {
    // 反向传播计算要用到
    input_cache_ = input.clone();
  }

  // 计算需要展平的维度
  const auto folded_dim = input.count(0, input.dim() - 1);
  // 计算输出形状
  auto output_shape = input.sizes();
  output_shape.pop_back();
  // 添加输出维度
  output_shape.push_back(W_.size(0));

  // output 必须清零，matmul_B_transpose 实现的是累积，不是直接赋值
  auto output = Tensor::zeros(input.dtype(), output_shape);

  // 对输入和输出重塑为 2D，以满足矩阵乘法的条件
  auto input_folded = input.reshape({folded_dim, input.sizes().back()});
  auto reshaped_out = output.reshape({folded_dim, output.sizes().back()});

  const auto M = input_folded.size(0);
  const auto N = W_.size(0);
  const auto K = W_.size(1);

  // Y = X @ W^T + b
  if (ctx()->device.is_blas()) {
    vec::matmul_B_transpose_blas(
        input_folded.const_data_as<float>(), input_folded.size(-1),
        W_.const_data_as<float>(), W_.size(-1),
        has_bias_ ? b_.const_data_as<float>() : nullptr,
        reshaped_out.mutable_data_as<float>(), reshaped_out.size(-1), M, N, K);
  } else {
    vec::matmul_B_transpose_native(
        input_folded.const_data_as<float>(), input_folded.size(-1),
        W_.const_data_as<float>(), W_.size(-1),
        has_bias_ ? b_.const_data_as<float>() : nullptr,
        reshaped_out.mutable_data_as<float>(), reshaped_out.size(-1), M, N, K);
  }

  return output;
}

Tensor Linear::backward(const Tensor &grad_output) {
  if (!grad_W_.is_defined()) {
    // grad_W_ 的形状: [out_features, in_features] (权重梯度矩阵)
    grad_W_ = Tensor::zeros_like(W_);
  }
  if (has_bias_) {
    if (!grad_b_.is_defined()) {
      // grad_b_ 的形状: [out_features] (偏置梯度向量)
      grad_b_ = Tensor::zeros_like(b_);
    }
  }

  auto input = input_cache_; // 前向时的输入

  // 反向梯度与输入形状相同,这个梯度要传给前一层
  auto grad_input = Tensor::zeros_like(input);

  // 计算需要展平的维度
  const auto folded_dim = grad_output.count(0, grad_output.dim() - 1);

  auto input_reshape = input.reshape({folded_dim, input.sizes().back()});
  auto grad_output_reshape =
      grad_output.reshape({folded_dim, grad_output.sizes().back()});
  auto grad_input_reshape =
      grad_input.reshape({folded_dim, grad_input.sizes().back()});

  // 计算对 W_ 的梯度，这里包含转置乘法，公式是: grad_W = grad_output^T @ X
  if (ctx()->device.is_blas()) {
    vec::matmul_A_transpose_blas(
        grad_output_reshape.const_data_as<float>(),
        grad_output_reshape.size(-1), input_reshape.const_data_as<float>(),
        input_reshape.size(-1), nullptr, grad_W_.mutable_data_as<float>(),
        grad_W_.size(-1), grad_output_reshape.size(0) /*K*/,
        grad_output_reshape.size(-1) /*M*/, grad_input_reshape.size(-1) /*N*/);
  } else {
    vec::matmul_A_transpose_native(
        grad_output_reshape.const_data_as<float>(),
        grad_output_reshape.size(-1), input_reshape.const_data_as<float>(),
        input_reshape.size(-1), nullptr, grad_W_.mutable_data_as<float>(),
        grad_W_.size(-1), grad_output_reshape.size(0) /*K*/,
        grad_output_reshape.size(-1) /*M*/, grad_input_reshape.size(-1) /*N*/);
  }

  if (has_bias_) {
    /* 计算对 b_ 的梯度，公式是: grad_b = sum(grad_output, axis=0)
       这里我们只需要对输出维度进行累加，假如 grad_output 形状[3,4]

        [1,2,3,4]
        [5,6,7,8]
        [9,10,11,12]

       grad_b[0] = 1+5+9  = 15
       grad_b[1] = 2+6+10 = 18
       grad_b[2] = 3+7+11 = 21
       grad_b[3] = 4+8+12 = 24
    */

    auto M = grad_output_reshape.size(0);
    auto N = grad_output_reshape.size(1);
    auto grad_bp = grad_b_.mutable_data_as<float>();
    auto grad_out_ptr = grad_output_reshape.const_data_as<float>();

    std::vector<float> ones(M, 1.0f);
    // 这里使用矩阵与向量的乘法
    if (ctx()->device.is_blas()) {
      vec::sgemv_transpose_blas(M, N, 1.0, grad_out_ptr, N, &ones[0], 1, 1.0,
                                grad_bp, 1);
    } else {
      // 将 grad_output_reshape 按列求和
      for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
          sum += grad_out_ptr[m * N + n];
        }
        grad_bp[n] += sum;
      }
    }
  }

  // 对 输入 X 的梯度计算
  // 公式是: grad_X = grad_output @ W_
  if (ctx()->device.is_blas()) {
    vec::matmul_blas(
        grad_output_reshape.const_data_as<float>(),
        grad_output_reshape.size(-1), W_.const_data_as<float>(), W_.size(-1),
        nullptr, grad_input_reshape.mutable_data_as<float>(),
        grad_input_reshape.size(-1), grad_output_reshape.size(0) /*M*/,
        grad_output_reshape.size(-1) /*K*/, grad_input_reshape.size(-1) /*N*/);
  } else {
    vec::matmul_native(
        grad_output_reshape.const_data_as<float>(),
        grad_output_reshape.size(-1), W_.const_data_as<float>(), W_.size(-1),
        nullptr, grad_input_reshape.mutable_data_as<float>(),
        grad_input_reshape.size(-1), grad_output_reshape.size(0) /*M*/,
        grad_output_reshape.size(-1) /*K*/, grad_input_reshape.size(-1) /*N*/);
  }

  return grad_input;
}

} // namespace dense