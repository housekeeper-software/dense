#ifndef LOSS_CROSS_ENTROPY_LOSS_H_
#define LOSS_CROSS_ENTROPY_LOSS_H_

#include "loss/loss.h"

namespace dense {

class CrossEntropyLoss : public Loss {
public:
  CrossEntropyLoss(int64_t ignore_index = -100);
  ~CrossEntropyLoss() = default;
  double forward(const dense::Tensor &input,
                 const dense::Tensor &target) override;
  dense::Tensor backward() override;

private:
  int64_t ignore_index_;
  dense::Tensor cached_one_hot_; // 缓存真实标签 (one-hot 编码)
  dense::Tensor cached_softmax_; // 缓存 Softmax 概率 (用于反向传播的简化)

  std::vector<bool> cached_ignored_mask_; // 记录被忽略的样本

  dense::Tensor z_shift_;
  int64_t num_ignored_;

  CrossEntropyLoss(const CrossEntropyLoss &) = delete;
  CrossEntropyLoss &operator=(const CrossEntropyLoss &) = delete;
};
} // namespace dense

#endif // LOSS_CROSS_ENTROPY_LOSS_H_