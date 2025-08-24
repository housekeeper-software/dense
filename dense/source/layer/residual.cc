#include "layer/residual.h"
#include "math/vec_math.h"

namespace dense {

Residual::Residual(Context *ctx, const std::string &name,
                   std::vector<std::unique_ptr<Layer>> layers)
    : Sequential(ctx, name, std::move(layers)) {}

dense::Tensor Residual::forward(const dense::Tensor &input) {
  auto shortcut = input.clone();

  auto output = Sequential::forward(input);

  auto shortcut_ptr = shortcut.const_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();

  if (ctx()->device.is_blas()) {
    vec::saxpy_blas(shortcut.numel(), 1.0f, shortcut_ptr, 1, out_ptr, 1);
  } else {
    vec::saxpy_native(shortcut.numel(), 1.0f, shortcut_ptr, 1, out_ptr, 1);
  }
  /* 等价
  for (size_t i = 0; i < shortcut.numel(); ++i) {
    out_ptr[i] += shortcut_ptr[i];
  }
  */
  return output;
}

dense::Tensor Residual::backward(const dense::Tensor &grad_output) {
  auto shortcut = grad_output.clone();

  auto grad_input = Sequential::backward(grad_output);

  auto shortcut_ptr = shortcut.const_data_as<float>();
  auto grad_in_ptr = grad_input.mutable_data_as<float>();

  if (ctx()->device.is_blas()) {
    vec::saxpy_blas(shortcut.numel(), 1.0f, shortcut_ptr, 1, grad_in_ptr, 1);
  } else {
    vec::saxpy_native(shortcut.numel(), 1.0f, shortcut_ptr, 1, grad_in_ptr, 1);
  }
  /* 等价
  for (size_t i = 0; i < shortcut.numel(); ++i) {
    grad_in_ptr[i] += shortcut_ptr[i];
  }*/
  return grad_input;
}
} // namespace dense