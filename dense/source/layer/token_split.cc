#include "layer/token_split.h"
#include "math/vec_math.h"

namespace dense {

TokenSplit::TokenSplit(Context *ctx, const std::string &name)
    : Layer(ctx, name) {}

Tensor TokenSplit::forward(const Tensor &input) {
  shape_ = input.sizes();
  const auto B = input.size(0);
  const auto T = input.size(1);
  const auto C = input.size(2);
  auto output = dense::Tensor::empty(input.dtype(), {B, C});
  auto out_ptr = output.mutable_data_as<float>();
  auto logits_ptr = input.const_data_as<float>();
  for (int64_t b = 0; b < B; ++b) {
    auto out_bt = out_ptr + b * C;
    auto logits_bt = logits_ptr + b * T * C;
    vec::scopy_blas(C, logits_bt, 1, out_bt, 1);
  }
  return output;
}

Tensor TokenSplit::backward(const Tensor &grad_output) {
  auto grad_input = Tensor::zeros(grad_output.dtype(), shape_);
  const auto B = grad_input.size(0);
  const auto T = grad_input.size(1);
  const auto C = grad_input.size(2);
  for (int64_t b = 0; b < B; ++b) {
    auto grad_out_bt = grad_output.const_data_as<float>() + b * C;
    auto grad_in_bt = grad_input.mutable_data_as<float>() + b * T * C;
    vec::scopy_blas(C, grad_out_bt, 1, grad_in_bt, 1);
  }
  return grad_input;
}
} // namespace dense