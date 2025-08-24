#ifndef OPTIM_OPTIMIZER_H_
#define OPTIM_OPTIMIZER_H_

#include "layer/layer.h"
#include <optional>

namespace dense {

class Optimizer {
public:
  Optimizer() = default;
  virtual ~Optimizer() = default;
  virtual void update(std::vector<ParamsAndGrads> &params_and_grads,
              std::optional<float> max_norm = std::nullopt) = 0;
  virtual double get_lr() const = 0;
  virtual void set_lr(const double lr) = 0;

private:
  Optimizer(const Optimizer &) = delete;
  Optimizer &operator=(const Optimizer &) = delete;
};

} // namespace dense

#endif // OPTIM_OPTIMIZER_H_