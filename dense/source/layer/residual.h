#ifndef LAYER_RESIDUAL_H_
#define LAYER_RESIDUAL_H_

#include "layer/layer.h"
#include "layer/sequential.h"
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace dense {

class Residual : public Sequential {
public:
  Residual(Context *ctx, const std::string &name,
           std::vector<std::unique_ptr<Layer>> layers);
  ~Residual() override = default;

  const char *type() const override { return "residual"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  Residual(const Residual &) = delete;
  Residual &operator=(const Residual &) = delete;
};

} // namespace dense

#endif // LAYER_RESIDUAL_H_