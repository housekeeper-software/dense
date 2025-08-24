#include "layer/relu.h"

namespace dense {

ReLU::ReLU(Context *ctx, const std::string &name, float negative_slope)
    : Layer(ctx, name), negative_slope_(negative_slope) {
  // 没有可学习参数
}

Tensor ReLU::forward(const Tensor &input) {
  if (is_training()) {
    input_cache_ = input.clone();
  }
  auto output = Tensor::zeros_like(input);

  auto in_ptr = input.const_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();

  for (size_t i = 0; i < input.numel(); ++i) {
    out_ptr[i] = (in_ptr[i] > 0) ? in_ptr[i] : negative_slope_ * in_ptr[i];
  }
  return output;
}

Tensor ReLU::backward(const Tensor &grad_output) {
  auto grad_input = Tensor::zeros_like(grad_output);

  auto grad_out_ptr = grad_output.const_data_as<float>();
  auto grad_in_ptr = grad_input.mutable_data_as<float>();
  auto in_ptr = input_cache_.const_data_as<float>();

  for (size_t i = 0; i < grad_input.numel(); ++i) {
    if (in_ptr[i] > 0) {
      grad_in_ptr[i] = grad_out_ptr[i];
    } else {
      grad_in_ptr[i] = grad_out_ptr[i] * negative_slope_;
    }
  }
  return grad_input;
}
} // namespace dense