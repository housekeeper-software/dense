#ifndef LAYER_FLATTEN_H_
#define LAYER_FLATTEN_H_

#include "layer/layer.h"
#include <memory>

namespace dense {

class Flatten : public Layer {
public:
  Flatten(Context *ctx, const std::string &name, int64_t start_axis,
          int64_t end_axis);
  ~Flatten() override = default;

  const char *type() const override { return "flatten"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  int64_t start_axis_;
  int64_t end_axis_;
  std::vector<int64_t> shape_;

  Flatten(const Flatten &) = delete;
  Flatten &operator=(const Flatten &) = delete;
};

} // namespace dense

#endif // LAYER_FLATTEN_H_