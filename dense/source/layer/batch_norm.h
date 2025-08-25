#ifndef LAYER_BATCH_NORM_H_
#define LAYER_BATCH_NORM_H_

#include "layer/layer.h"

namespace dense {

// BatchNorm1D, BatchNorm2D, BatchNorm3D
// 只能在推理时使用，完全 CPU 计算
// input:
//       BatchNorm1D,形状 [N, C] or [N, C, L], 'L' is the sequence length
//       BatchNorm2D,形状 [N, C, H, W]
//       BatchNorm3D,形状 [N, C, D, H, W]
// running_mean: 滑动均值，在训练过程中，所有批次的均值的指数移动均值
// running_var: 滑动方差，在训练过程中所有批次的方差的指数移动均值
// gamma: 缩放因子，训练所得
// beta: 偏移量，训练所得
// eps: 小常数，防止除零，一般模型配置中会提供此值
// affine: 是否需要仿射变换
// track_running_stats：是否启动 running_mean ,running_var

class BatchNorm : public Layer {
public:
  BatchNorm(Context *ctx, const std::string &name, int64_t num_features,
            float eps = 1e-5, float momentum = 0.1, bool affine = true,
            bool track_running_stats = true);

  ~BatchNorm() override = default;
  const char *type() const override { return "bn"; }

  void init() override;

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
  forward_device(const Tensor &input);

  int64_t num_features_;     // 输入通道数
  float eps_;                // 极小值，避免除以零
  float momentum_;           // 动量，用于滑动平均
  bool affine_;              // 是否使用可学习参数进行仿射变换
  bool track_running_stats_; // 是否跟踪运行时均值和方差

  Tensor mean_, x_norm_, rstd_; // 用于反向传播

  BatchNorm(const BatchNorm &) = delete;
  BatchNorm &operator=(const BatchNorm &) = delete;
};

} // namespace dense

#endif // LAYER_BATCH_NORM_H_