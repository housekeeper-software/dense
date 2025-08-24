#include "layer/gelu.h"
#include "layer/init.h"
#include <cmath>

namespace dense {

namespace {
#define GELU_SCALING_FACTOR std::sqrt(2.0f / M_PI)
} // namespace

GeLU::GeLU(Context *ctx, const std::string &name) : Layer(ctx, name) {}

dense::Tensor GeLU::forward(const dense::Tensor &input) {
  // GELU 激活函数的公式是: x * P(X <= x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x
  // + 0.044715 * x^3)))
  if (is_training()) {
    input_cache_ = input.clone();
  }

  auto N = input.numel();

  auto output = dense::Tensor::zeros_like(input);

  auto in_ptr = input.const_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();

  for (size_t i = 0; i < N; ++i) {
    float x = in_ptr[i];
    float cube = 0.044715f * x * x * x;
    out_ptr[i] =
        0.5f * x * (1.0f + std::tanh(GELU_SCALING_FACTOR * (x + cube)));
  }
  return output;
}

dense::Tensor GeLU::backward(const dense::Tensor &grad_output) {
  // GELU 激活函数的导数计算：GELU'(x)
  // GELU'(x) = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) +
  //             0.5 * x * (1 - tanh^2(sqrt(2/pi) * (x + 0.044715 * x^3))) *
  //             sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
  auto input = input_cache_;

  auto grad_input = dense::Tensor::zeros_like(grad_output);

  auto N = input.numel();

  auto grad_output_ptr = grad_output.const_data_as<float>();
  auto in_ptr = input.const_data_as<float>();

  auto grad_input_ptr = grad_input.mutable_data_as<float>();

  for (size_t i = 0; i < N; ++i) {
    float x = in_ptr[i];

    float u = GELU_SCALING_FACTOR * (x + 0.044715f * x * x * x);
    float du_dx = GELU_SCALING_FACTOR * (1.0 + 0.044715f * 3.0 * x * x);
    float tanh_val = std::tanh(u);

    auto dg_dx =
        0.5 * (1 + tanh_val) + 0.5 * x * (1 - tanh_val * tanh_val) * du_dx;
    grad_input_ptr[i] = dg_dx * grad_output_ptr[i];
  }
  return grad_input;
}

} // namespace dense