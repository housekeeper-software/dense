#include "layer/dropout.h"
#include "layer/init.h"
#include "math/vec_math.h"

namespace dense {

Dropout::Dropout(Context *ctx, const std::string &name, float dropout_ratio)
    : Layer(ctx, name), dropout_ratio_(dropout_ratio),
      scale_(1.0 / (1.0 - dropout_ratio)) {}

dense::Tensor Dropout::forward(const dense::Tensor &input) {
  if (!is_training())
    return input.clone(); // 更加安全

  auto mask = dense::Tensor::empty(input.dtype(), input.sizes());
  init::uniform_(mask);
  return forward_with_mask(input, mask);
}

Tensor Dropout::forward_with_mask(const Tensor &input, const Tensor &mask) {
  mask_ = mask;

  auto mask_ptr = mask_.mutable_data_as<float>();

  for (size_t i = 0; i < input.numel(); ++i) {
    // 等价于 * scale_
    mask_ptr[i] = (mask_ptr[i] > dropout_ratio_) ? scale_ : 0.0f;
  }

  auto output = dense::Tensor::zeros_like(input);

  auto in_ptr = input.const_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();

  // 这里用哈达玛积
  if (ctx()->device.is_blas()) {
    vec::shdm_blas(input.numel(), in_ptr, mask_ptr, out_ptr);
  } else {
    for (size_t i = 0; i < input.numel(); ++i) {
      out_ptr[i] = in_ptr[i] * mask_ptr[i];
    }
  }
  return output;
}

dense::Tensor Dropout::backward(const dense::Tensor &grad_output) {
  auto grad_input = dense::Tensor::zeros_like(grad_output);

  auto mask_ptr = mask_.const_data_as<float>();
  auto grad_out_ptr = grad_output.const_data_as<float>();
  auto grad_in_ptr = grad_input.mutable_data_as<float>();

  if (ctx()->device.is_blas()) {
    vec::shdm_blas(grad_output.numel(), grad_out_ptr, mask_ptr, grad_in_ptr);
  } else {
    for (size_t i = 0; i < grad_output.numel(); ++i) {
      grad_in_ptr[i] = grad_out_ptr[i] * mask_ptr[i];
    }
  }
  return grad_input;
}
} // namespace dense