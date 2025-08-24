#ifndef LAYER_GELU_H_
#define LAYER_GELU_H_

#include "layer/layer.h"
#include <memory>

namespace dense {

class GeLU : public Layer {
public:
  GeLU(Context *ctx, const std::string &name);
  ~GeLU() override = default;

  const char *type() const override { return "gelu"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  dense::Tensor input_cache_;

  GeLU(const GeLU &) = delete;
  GeLU &operator=(const GeLU &) = delete;
};

} // namespace dense

#endif // LAYER_GELU_H_