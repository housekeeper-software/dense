#ifndef LAYER_RELU_H_
#define LAYER_RELU_H_

#include "layer/layer.h"

namespace dense {

// ReLU: negative_slope = 0
// Leaky ReLU: negative_slope = 0.01 (固定小值)
class ReLU : public Layer {
public:
  ReLU(Context *ctx, const std::string &name, float negative_slope = 0.0f);
  ~ReLU() override = default;
  const char *type() const override { return "relu"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  float negative_slope_;
  Tensor input_cache_;
  ReLU(const ReLU &) = delete;
  ReLU &operator=(const ReLU &) = delete;
};

} // namespace dense

#endif // LAYER_RELU_H_