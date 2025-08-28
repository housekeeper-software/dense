#ifndef LAYER_DROP_PATH_H_
#define LAYER_DROP_PATH_H_

#include "layer/layer.h"

namespace dense {

/*
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
Stochastic Depth
是一种正则化技术，主要用于深度神经网络，特别是在Vision
Transformer和ResNet等架构中

适合使用DropPath的场景：
1. 深度网络（>20层）
2. 残差连接架构（ResNet, DenseNet）
3. Transformer架构（ViT, BERT等）
4. 训练时间较长的大模型
5. 容易过拟合的任务

- 小模型：max_drop_prob = 0.1-0.2
- 中等模型：max_drop_prob = 0.2-0.3
- 大模型：max_drop_prob = 0.3-0.5
*/

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