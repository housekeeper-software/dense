#ifndef LAYER_SEQUENTIAL_H_
#define LAYER_SEQUENTIAL_H_

#include "layer/layer.h"
#include <memory>

namespace dense {

class Sequential : public Layer {
public:
  Sequential(Context *ctx, const std::string &name,
             std::vector<std::unique_ptr<Layer>> layers);
  virtual ~Sequential() override = default;

  virtual const char *type() const override { return "sequential"; }

  virtual Tensor forward(const Tensor &input) override;
  virtual Tensor backward(const Tensor &grad_output) override;

private:
  std::vector<std::unique_ptr<Layer>> layers_;

  Sequential(const Sequential &) = delete;
  Sequential &operator=(const Sequential &) = delete;
};

} // namespace dense

#endif // LAYER_SEQUENTIAL_H_