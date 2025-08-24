#ifndef LAYER_LINEAR_H_
#define LAYER_LINEAR_H_

#include "layer/init.h"
#include "layer/layer.h"

namespace dense {

// 线性层，执行 Y=X@W^T+b
class Linear : public Layer {
public:
  Linear(Context *ctx, const std::string &name, int64_t in_features,
         int64_t out_features, bool has_bias = true);
  ~Linear() override = default;
  const char *type() const override { return "linear"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  int64_t in_features_;
  int64_t out_features_;
  bool has_bias_;

  Tensor input_cache_;

  Linear(const Linear &) = delete;
  Linear &operator=(const Linear &) = delete;
};

} // namespace dense

#endif // LAYER_LINEAR_H_