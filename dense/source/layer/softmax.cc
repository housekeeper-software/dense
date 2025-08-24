#include "layer/softmax.h"
#include "math/vec_math.h"

namespace dense {

Softmax::Softmax(Context *ctx, const std::string &name) : Layer(ctx, name) {}

Tensor Softmax::forward(const Tensor &input) {
  const auto folded_dim = input.count(0, input.dim() - 1);
  auto reshape_input = input.reshape({folded_dim, input.sizes().back()});
  auto C = input.size(-1);

  auto output = input.clone();
  auto reshape_output = output.reshape({folded_dim, output.sizes().back()});
  if (ctx()->device.is_blas()) {
    vec::mat_softmax_forward_blas(reshape_output.mutable_data_as<float>(),
                                  folded_dim, C);
  } else {
    vec::mat_softmax_forward_native(reshape_output.mutable_data_as<float>(),
                                    folded_dim, C);
  }
  if (is_training()) {
    softmax_output_ = output;
  }
  return output;
}

Tensor Softmax::backward(const Tensor &grad_output) {
  const auto folded_dim = grad_output.count(0, grad_output.dim() - 1);
  auto reshape_grad_output =
      grad_output.reshape({folded_dim, grad_output.sizes().back()});
  auto C = grad_output.size(-1);

  auto grad_input = Tensor::zeros_like(grad_output);
  auto reshape_grad_input =
      grad_input.reshape({folded_dim, grad_input.sizes().back()});
  auto reshape_softmax_output =
      softmax_output_.reshape({folded_dim, softmax_output_.sizes().back()});
      
  if (ctx()->device.is_blas()) {
    vec::mat_softmax_backward_blas(
        reshape_grad_input.mutable_data_as<float>(),
        reshape_softmax_output.const_data_as<float>(),
        reshape_grad_output.const_data_as<float>(), folded_dim, C);
  } else {
    vec::mat_softmax_backward_native(
        reshape_grad_input.mutable_data_as<float>(),
        reshape_softmax_output.const_data_as<float>(),
        reshape_grad_output.const_data_as<float>(), folded_dim, C);
  }
  return grad_input;
}

} // namespace dense