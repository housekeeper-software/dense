#include "layer/sequential.h"

namespace dense {

Sequential::Sequential(Context *ctx, const std::string &name,
                       std::vector<std::unique_ptr<Layer>> layers)
    : Layer(ctx, name), layers_(std::move(layers)) {}

dense::Tensor Sequential::forward(const dense::Tensor &input) {
  dense::Tensor x = input;
  for (auto &i : layers_) {
    x = i->forward(x);
  }
  return x;
}

dense::Tensor Sequential::backward(const dense::Tensor &grad_output) {
  dense::Tensor grad_input = grad_output;
  for (auto rit = layers_.rbegin(); rit != layers_.rend(); ++rit) {
    grad_input = (*rit)->backward(grad_input);
  }
  return grad_input;
}

} // namespace dense