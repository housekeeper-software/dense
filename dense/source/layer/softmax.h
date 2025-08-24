#ifndef LAYER_SOFTMAX_H_
#define LAYER_SOFTMAX_H_

#include "layer/layer.h"
#include <memory>

namespace dense {

class Softmax : public Layer {
public:
  Softmax(Context *ctx, const std::string &name);
  ~Softmax() override = default;

  const char *type() const  override { return "softmax"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  dense::Tensor softmax_output_;

  Softmax(const Softmax &) = delete;
  Softmax &operator=(const Softmax &) = delete;
};

} // namespace dense

#endif // LAYER_SOFTMAX_H_