#ifndef LAYER_DROPOUT_H_
#define LAYER_DROPOUT_H_

#include "layer/layer.h"
#include <memory>

namespace dense {

class Dropout : public Layer {
public:
  Dropout(Context *ctx, const std::string &name, float dropout_ratio = 0.5);
  ~Dropout() = default;

  const char *type() const override { return "dropout"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  float dropout_ratio_; // 设置为零的神经元比例（丢弃率）
  float scale_;         // 缩放因子，用于 inverted dropout
  dense::Tensor mask_;  // 缓存丢弃掩码，用于反向传播

  Dropout(const Dropout &) = delete;
  Dropout &operator=(const Dropout &) = delete;
};

} // namespace dense

#endif // LAYER_DROPOUT_H_