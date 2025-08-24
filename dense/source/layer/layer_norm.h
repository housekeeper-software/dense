#ifndef LAYER_LAYER_NOTMAL_H_
#define LAYER_LAYER_NOTMAL_H_

#include "layer/layer.h"
#include <memory>
#include <string>

namespace dense {

// 层归一化
// elementwise_affine: 是否使用可学习参数进行仿射变换
// has_bias: 是否有偏置项
// normalized_shape: 归一化的形状
// epsilon: 防止除以零的极小值

class LayerNorm : public Layer {
public:
  LayerNorm(Context *ctx, const std::string &name,
            const std::vector<int64_t> &normalized_shape, float epsilon = 1e-05,
            bool elementwise_affine = true, bool has_bias = true);
  ~LayerNorm() override = default;

  const char *type() const override { return "ln"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  dense::Tensor forward_cpu(const dense::Tensor &input);
  dense::Tensor forward_blas(const dense::Tensor &input);
  dense::Tensor backward_cpu(const dense::Tensor &grad_output);
  dense::Tensor backward_blas(const dense::Tensor &grad_output);

  std::vector<int64_t> normalized_shape_;
  float epsilon_;
  bool elementwise_affine_;
  bool has_bias_;

  Tensor mean_, x_norm_, rstd_; // 用于反向传播

  dense::Tensor hdm_; // 用于存储哈达玛积结果

  LayerNorm(const LayerNorm &) = delete;
  LayerNorm &operator=(const LayerNorm &) = delete;
};

} // namespace dense

#endif // LAYER_LAYER_NOTMAL_H_