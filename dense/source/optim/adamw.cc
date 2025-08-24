#include "optim/adamw.h"

namespace dense {

AdamW::AdamW(double learning_rate, double beta1, double beta2, double epsilon,
             double weight_decay)
    : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      weight_decay_(weight_decay), step_(0) {}

void AdamW::ensure_state(int group_idx, int param_idx,
                         const dense::Tensor &param) {
  // 扩展容器大小
  if (m_states_.size() <= group_idx) {
    m_states_.resize(group_idx + 1);
    v_states_.resize(group_idx + 1);
  }
  if (m_states_[group_idx].size() <= param_idx) {
    m_states_[group_idx].resize(param_idx + 1);
    v_states_[group_idx].resize(param_idx + 1);
  }

  // 惰性初始化
  if (!m_states_[group_idx][param_idx].is_defined()) {
    m_states_[group_idx][param_idx] = dense::Tensor::zeros_like(param);
    v_states_[group_idx][param_idx] = dense::Tensor::zeros_like(param);
  }
}

void AdamW::update(std::vector<ParamsAndGrads> &params_and_grads,
                   std::optional<float> max_norm) {
  if (max_norm && max_norm.has_value()) {
    clip_gradients(params_and_grads, max_norm.value());
  }
  ++step_;

  // 预计算偏差修正因子
  double bias_correction1 = 1.0 - std::pow(beta1_, step_);
  double bias_correction2 = 1.0 - std::pow(beta2_, step_);

  for (size_t i = 0; i < params_and_grads.size(); ++i) {
    auto params = params_and_grads[i].params;
    auto grads = params_and_grads[i].grads;

    for (size_t j = 0; j < grads.size(); ++j) {
      auto param = params[j];
      auto grad = grads[j];

      if (param.is_defined() && grad.is_defined() && param.numel() > 0) {
        ensure_state(i, j, param);
        size_t N = param.numel();
        auto param_ptr = param.mutable_data_as<float>();
        auto grad_ptr = grad.mutable_data_as<float>();
        if (j == 0 && weight_decay_ != 0) {
          // 等价  p.mul_(1 - options.lr() * options.weight_decay());
          for (size_t k = 0; k < N; ++k) {
            param_ptr[k] -= lr_ * weight_decay_ * param_ptr[k];
            // param_ptr[i] *= (1 - lr_ * weight_decay_);
          }
          //*param *= (1 - lr_ * weight_decay_);
        }

        auto &m = m_states_[i][j];
        auto &v = v_states_[i][j];
        auto m_ptr = m.mutable_data_as<float>();
        auto v_ptr = v.mutable_data_as<float>();
        for (size_t k = 0; k < N; ++k) {
          m_ptr[k] = beta1_ * m_ptr[k] + (1 - beta1_) * grad_ptr[k];
          v_ptr[k] =
              beta2_ * v_ptr[k] + (1 - beta2_) * (grad_ptr[k] * grad_ptr[k]);
          auto m_hat = m_ptr[k] / bias_correction1;
          auto v_hat = v_ptr[k] / bias_correction2;
          auto adam_update = m_hat / (std::sqrt(v_hat) + epsilon_);
          param_ptr[k] -= lr_ * adam_update;
        }
        // 等价 param->addcdiv_(m_hat, v_hat.sqrt().add_(epsilon_), -lr_);
      }
    }
  }
}

void AdamW::clip_gradients(std::vector<ParamsAndGrads> &params_and_grads,
                           float max_norm) {
  if (max_norm <= 0.0f) {
    return;
  }

  auto calculate_norm = [](dense::Tensor &grad) -> double {
    double norm = 0.0f;
    auto data = grad.const_data_as<float>();
    for (size_t i = 0; i < grad.numel(); ++i) {
      norm += static_cast<double>(data[i] * data[i]);
    }
    return norm;
  };

  auto scale_grad = [](dense::Tensor &grad, double clip_coeff) {
    float *data = grad.mutable_data_as<float>();
    for (size_t i = 0; i < grad.numel(); ++i) {
      data[i] *= clip_coeff;
    }
  };

  double total_norm_sq = 0.0;

  for (auto &grad_group : params_and_grads) {
    for (auto &i : grad_group.grads) {
      if (i.is_defined()) {
        total_norm_sq += calculate_norm(i);
      }
    }
  }
  double total_norm = std::sqrt(total_norm_sq);

  if (total_norm > max_norm) {
    double clip_coeff = max_norm / (total_norm + 1e-6);

    for (auto &grad_group : params_and_grads) {
      for (auto &i : grad_group.grads) {
        if (i.is_defined()) {
          scale_grad(i, clip_coeff);
        }
      }
    }
  }
}

} // namespace dense