#ifndef LAYER_POOLING_H_
#define LAYER_POOLING_H_

#include "layer/layer.h"
#include <memory>

namespace dense {

enum PoolMethod { kMax = 0, kAvg };

class Pooling : public Layer {
public:
  Pooling(Context *ctx, const std::string &name, int pool_method,
          int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
          int64_t stride_w, int64_t pad_h, int64_t pad_w);
  ~Pooling() override = default;
  const char *type() const override { return "pooling"; }
  dense::Tensor indices() const { return max_idx_; }
  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  int method_;
  int64_t kernel_h_;
  int64_t kernel_w_;
  int64_t stride_h_;
  int64_t stride_w_;
  int64_t pad_h_;
  int64_t pad_w_;

  dense::Tensor max_idx_;

  std::vector<int64_t> input_shape_;

  Pooling(const Pooling &) = delete;
  Pooling &operator=(const Pooling &) = delete;
};

} // namespace dense

#endif // LAYER_POOLING_H_