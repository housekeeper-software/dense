#ifndef OPTM_ADAMW_H_
#define OPTM_ADAMW_H_

#include "optim/optimizer.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace dense {

class AdamW : public Optimizer {
public:
  // 构造函数：初始化优化器参数
  AdamW(double learning_rate = 1e-4, double beta1 = 0.9, double beta2 = 0.999,
        double epsilon = 1e-8, double weight_decay = 0.01);

  // 更新模型参数的方法
  // params_and_grads 包含了所有可训练参数及其梯度
  void update(std::vector<ParamsAndGrads> &params_and_grads,
              std::optional<float> max_norm = std::nullopt) override;

  void set_lr(double lr) override { lr_ = lr; }
  double get_lr() const override { // 可以添加一个获取当前学习率的方法
    return lr_;
  }

private:
  void ensure_state(int group_idx, int param_idx, const dense::Tensor &param);

  void clip_gradients(std::vector<ParamsAndGrads> &params_and_grads,
                      float max_norm);

  double lr_;
  double beta1_;
  double beta2_;
  double epsilon_;
  double weight_decay_;
  int step_; // 时间步，用于偏差修正

  // 存储每个参数的第一次矩估计 (m)
  std::vector<std::vector<dense::Tensor>> m_states_;
  // 存储每个参数的第二次矩估计 (v)
  std::vector<std::vector<dense::Tensor>> v_states_;

  AdamW(const AdamW &) = delete;
  AdamW &operator=(const AdamW &) = delete;
};
} // namespace dense

#endif // OPTM_ADAMW_H_