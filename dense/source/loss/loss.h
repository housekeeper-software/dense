#ifndef LOSS_LOSS_H_
#define LOSS_LOSS_H_

#include "base/tensor.h"

namespace dense {

class Loss {
public:
  Loss() = default;
  virtual ~Loss() = default;
  virtual double forward(const dense::Tensor &input,
                         const dense::Tensor &target) = 0;
  virtual dense::Tensor backward() = 0;

private:
  Loss(const Loss &) = delete;
  Loss &operator=(const Loss &) = delete;
};
} // namespace dense

#endif // LOSS_LOSS_H_