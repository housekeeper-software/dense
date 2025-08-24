#include "loss/cross_entropy_loss.h"
#include <cmath>
#include <stdexcept>

namespace dense {
namespace {
// 检查target是否为有效的类别索引
bool is_class_indices_target(const dense::Tensor &input,
                             const dense::Tensor &target) {
  // 1. 数据类型检查：类别索引必须是整数类型
  if (target.dtype() != dense::DType::kInt64 &&
      target.dtype() != dense::DType::kInt32) {
    return false;
  }

  // 2. 形状检查：类别索引的维度应该比input少1维
  // input: (N, C) 或 (N, C, d1, d2, ..., dk)
  // target: (N) 或 (N, d1, d2, ..., dk) - 缺少类别维度C
  if (target.dim() != input.dim() - 1) {
    return false;
  }

  // 3. 除了最后一维(类别维度)，其他维度大小必须匹配
  for (int i = 0; i < target.dim(); ++i) {
    if (target.size(i) != input.size(i)) {
      return false;
    }
  }

  return true;
}

// 检查target是否为有效的概率分布
bool is_probabilities_target(const dense::Tensor &input,
                             const dense::Tensor &target) {
  // 1. 数据类型检查：概率必须是浮点类型
  if (target.dtype() != dense::DType::kFloat32 &&
      target.dtype() != dense::DType::kFloat64) {
    return false;
  }

  // 2. 形状检查：概率分布必须与input形状完全相同
  if (target.sizes() != input.sizes()) {
    return false;
  }

  return true;
}
} // namespace

CrossEntropyLoss::CrossEntropyLoss(int64_t ignore_index)
    : ignore_index_(ignore_index), num_ignored_(0) {}

double CrossEntropyLoss::forward(const dense::Tensor &input,
                                 const dense::Tensor &target) {
  if (input.dim() < 2) {
    throw std::runtime_error("The expected input should be at least 2D.");
  }
  auto C = input.size(-1);

  const auto folded_dim = input.count(0, input.dim() - 1);

  num_ignored_ = 0;

  cached_ignored_mask_ = std::vector<bool>(folded_dim, false);

  auto squeezed_target = target.squeeze(-1);

  if (is_class_indices_target(input, squeezed_target)) {
    // 如果 target 是整数标签，转换为 one-hot 编码
    // 创建一个全零的张量，作为 one-hot 编码的容器
    auto one_hot = dense::Tensor::zeros_like(input);
    one_hot = one_hot.reshape({folded_dim, C});
    auto reshaped_target = squeezed_target.reshape({folded_dim});
    for (size_t b = 0; b < folded_dim; ++b) {
      int64_t target_class;
      if (target.dtype() == DType::kInt64) {
        target_class = reshaped_target.const_data_as<int64_t>()[b];
      } else {
        target_class =
            static_cast<int64_t>(reshaped_target.const_data_as<int32_t>()[b]);
      }
      if (target_class == ignore_index_) {
        // 如果是ignore_index，跳过这个样本
        cached_ignored_mask_[b] = true;
        ++num_ignored_;
        continue;
      }
      if (target_class < 0 || target_class >= static_cast<int64_t>(C)) {
        throw std::runtime_error("Class index out of range [0, " +
                                 std::to_string(C) + ")");
      }

      auto one_hot_bt = one_hot.mutable_data_as<float>() + b * C;
      one_hot_bt[target_class] = 1.0f;
    }
    cached_one_hot_ = one_hot;
  } else if (is_probabilities_target(input, squeezed_target)) {
    cached_one_hot_ = squeezed_target;
  } else {
    throw std::runtime_error("Target format not supported.");
  }

  if (num_ignored_ == static_cast<int64_t>(folded_dim)) {
    throw std::runtime_error("All samples are ignored, cannot compute loss");
  }

  cached_softmax_ = Tensor::zeros_like(input);
  double total_loss = 0.0;

  auto reshape_input = input.reshape({folded_dim, C});
  auto reshape_one_hot = cached_one_hot_.reshape({folded_dim, C});
  auto reshape_softmax = cached_softmax_.reshape({folded_dim, C});

  if (!z_shift_.is_defined() || z_shift_.size(0) != C) {
    z_shift_ = Tensor::empty(input.dtype(), {C});
  }

  auto z_shift_ptr = z_shift_.mutable_data_as<float>();

  for (size_t b = 0; b < folded_dim; ++b) {
    // 跳过被忽略的样本
    if (cached_ignored_mask_[b]) {
      continue;
    }

    auto offset = b * C;
    auto in_bt = reshape_input.const_data_as<float>() + offset;
    auto softmax_bt = reshape_softmax.mutable_data_as<float>() + offset;
    auto one_hot_bt = reshape_one_hot.const_data_as<float>() + offset;

    // 计算 z_max
    float z_max = -INFINITY;
    for (size_t k = 0; k < C; ++k) {
      if (in_bt[k] > z_max) {
        z_max = in_bt[k];
      }
    }
    // exp_sum= sum(e^(z_k-z_max))
    float exp_sum = 0.0f;
    for (size_t k = 0; k < C; ++k) {
      // z_k-z_max
      z_shift_ptr[k] = in_bt[k] - z_max;
      // e^(z_k-z_max)
      exp_sum += std::exp(z_shift_ptr[k]);
    }

    // log_sum_exp = log(sum(e^(z_k-z_max)))
    float log_sum_exp = std::log(exp_sum);

    double sum_of_products = 0.0f;
    for (size_t k = 0; k < C; ++k) {
      // [z_k-z_max - log(sum(e^(z_k-z_max)))]
      auto pred_log_softmax = z_shift_ptr[k] - log_sum_exp;
      // softmax_bt: pred_log_softmax 再取指数就还原为 softmax 值
      // 所以 softmax_bt 存储的是 Softmax 计算值，用于反向传播
      softmax_bt[k] = std::exp(pred_log_softmax);
      // sum(y_k* (pred_log_softmax))
      sum_of_products += one_hot_bt[k] * pred_log_softmax;
    }
    total_loss += sum_of_products;
  }
  // 损失函数 J = - sum(y_k* (pred_log_softmax))
  total_loss = -total_loss / static_cast<double>(folded_dim - num_ignored_);
  return total_loss;
}

dense::Tensor CrossEntropyLoss::backward() {
  auto grad = dense::Tensor::zeros_like(cached_softmax_);
  const auto folded_dim = cached_softmax_.count(0, cached_softmax_.dim() - 1);

  auto C = cached_softmax_.size(-1);

  auto reshape_grad = grad.reshape({folded_dim, C});
  auto reshape_softmax = cached_softmax_.reshape({folded_dim, C});
  auto reshape_one_hot = cached_one_hot_.reshape({folded_dim, C});

  int64_t num_valid = folded_dim - num_ignored_;

  for (size_t b = 0; b < folded_dim; ++b) {
    // 被忽略的样本梯度保持为0
    if (cached_ignored_mask_[b]) {
      // grad 已经初始化为0，无需额外操作
      continue;
    }

    auto offset = b * C;
    auto softmax_bt = reshape_softmax.const_data_as<float>() + offset;
    auto one_hot_bt = reshape_one_hot.const_data_as<float>() + offset;
    auto grad_bt = reshape_grad.mutable_data_as<float>() + offset;

    for (size_t k = 0; k < C; ++k) {
      // 计算梯度
      grad_bt[k] =
          (softmax_bt[k] - one_hot_bt[k]) / static_cast<double>(num_valid);
    }
  }
  return grad;
}
} // namespace dense
