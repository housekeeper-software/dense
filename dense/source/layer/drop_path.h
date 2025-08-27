#ifndef LAYER_DROP_PATH_H_
#define LAYER_DROP_PATH_H_

#include "layer/layer.h"

namespace dense {

class DropPath : public Layer {
public:
  DropPath(Context *ctx, const std::string &name, float drop_prob = 0.0f);
  ~DropPath() override = default;
  const char *type() const override { return "drop_path"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  float drop_prob_; // 丢弃概率
  dense::Tensor drop_mask_;
  DropPath(const DropPath &) = delete;
  DropPath &operator=(const DropPath &) = delete;
};

} // namespace dense

#endif // LAYER_DROP_PATH_H_