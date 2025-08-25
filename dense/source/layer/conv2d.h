#ifndef LAYER_CONV2D_H_
#define LAYER_CONV2D_H_

#include "layer/layer.h"

namespace dense {

class Conv2d : public Layer {
public:
  Conv2d(Context *ctx, const std::string &name, int64_t in_channels,
         int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
         int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
         bool has_bias = true);
  ~Conv2d() override;

  const char *type() const override { return "conv2d"; }

  void init() override;

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  class Conv2dHelper;
  class ConvTranspose2dHelper;

  int64_t in_channels_;
  int64_t out_channels_;
  int64_t kernel_h_;
  int64_t kernel_w_;

  int64_t stride_h_;
  int64_t stride_w_;
  int64_t pad_h_;
  int64_t pad_w_;
  bool has_bias_;
  std::vector<int64_t> in_shape_; // 缓存输入形状

  std::unique_ptr<Conv2dHelper> conv2d_;
  std::unique_ptr<ConvTranspose2dHelper> conv_transpose_2d_;

  Conv2d(const Conv2d &) = delete;
  Conv2d &operator=(const Conv2d &) = delete;
};

} // namespace dense

#endif // LAYER_CONV2D_H_