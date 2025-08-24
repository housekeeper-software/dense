#ifndef OPTM_COSINE_WARM_RESTARTS_H_
#define OPTM_COSINE_WARM_RESTARTS_H_

#include "optim/lr_scheduler.h"

namespace dense {

class CosineAnnealingWarmRestarts : public LRScheduler {
public:
  CosineAnnealingWarmRestarts(double initial_lr, int T_0, int T_mult = 1,
                              double eta_min = 0.0, int last_epoch = -1);
  ~CosineAnnealingWarmRestarts() override = default;
  void step(std::optional<double> epoch = std::nullopt) override;
  double get_lr() const override;

private:
  int T_0_;    // 初始周期长度
  int T_mult_; // 周期长度乘数
  double
      T_cur_; // 当前周期的 epoch 计数，注意这里依然使用 int，因为它是步进计数
  double eta_min_;    // 最小学习率
  double initial_lr_; // 原始初始学习率
  double last_epoch_; // 总的 epoch 计数，改为 double 以支持浮点数
  int T_i_;           // 当前周期的总长度

  CosineAnnealingWarmRestarts(const CosineAnnealingWarmRestarts &) = delete;
  CosineAnnealingWarmRestarts &
  operator=(const CosineAnnealingWarmRestarts &) = delete;
};
} // namespace dense

#endif // OPTM_COSINE_WARM_RESTARTS_H_