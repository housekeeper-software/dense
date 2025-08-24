#include "optim/cosine_warm_restarts.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace dense {

CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(double initial_lr,
                                                         int T_0, int T_mult,
                                                         double eta_min,
                                                         int last_epoch)
    : T_0_(T_0), T_mult_(T_mult), eta_min_(eta_min), initial_lr_(initial_lr),
      T_cur_(last_epoch), last_epoch_(last_epoch), T_i_(T_0) {

  if (T_0 <= 0) {
    throw std::invalid_argument("T_0 必须是正整数.");
  }
  if (T_mult < 1) {
    throw std::invalid_argument("T_mult 必须是大于等于 1 的整数.");
  }
  step();
}

void CosineAnnealingWarmRestarts::step(std::optional<double> epoch) {
  double current_epoch;

  if (!epoch.has_value() && last_epoch_ < 0)
    current_epoch = 0;

  if (!epoch.has_value()) {
    current_epoch = last_epoch_ + 1;
    T_cur_ = T_cur_ + 1;
    if (T_cur_ >= T_i_) {
      T_cur_ = T_cur_ - T_i_;
      T_i_ = T_i_ * T_mult_;
    }
  } else {
    current_epoch = epoch.value();

    if (current_epoch < 0) {
      throw std::invalid_argument("Epoch 必须是非负数.");
    }
    if (current_epoch >= T_0_) {
      if (T_mult_ == 1) {
        T_cur_ = fmod(current_epoch, T_0_);
      } else {
        double term = current_epoch / T_0_ * (T_mult_ - 1) + 1;
        double log_base_T_mult =
            std::log(term) / std::log(static_cast<double>(T_mult_));
        int n = static_cast<int>(std::floor(log_base_T_mult));

        double sum_of_previous_cycles =
            static_cast<double>(T_0_) *
            (static_cast<double>(std::pow(static_cast<double>(T_mult_), n)) -
             1) /
            (T_mult_ - 1);

        T_cur_ = current_epoch - static_cast<double>(sum_of_previous_cycles);
        T_i_ = T_0_ * static_cast<int>(std::pow(T_mult_, n));
      }
    } else {
      T_i_ = T_0_;
      T_cur_ = current_epoch;
    }
  }
  last_epoch_ = std::floor(current_epoch);
}

double CosineAnnealingWarmRestarts::get_lr() const {
  if (last_epoch_ < 0) {
    return initial_lr_;
  }

  // 使用余弦退火公式计算学习率
  return eta_min_ +
         (initial_lr_ - eta_min_) *
             (1 + std::cos(M_PI * static_cast<double>(T_cur_) /
                           static_cast<double>(T_i_))) /
             2.0;
}
} // namespace dense