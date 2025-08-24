#ifndef OPTIM_LR_SCHEDULER_H_
#define OPTIM_LR_SCHEDULER_H_

#include <optional>

namespace dense {

class LRScheduler {
public:
  LRScheduler() = default;
  virtual ~LRScheduler() = default;
  virtual void step(std::optional<double> epoch = std::nullopt) = 0;
  virtual double get_lr() const = 0;

private:
  LRScheduler(const LRScheduler &) = delete;
  LRScheduler &operator=(const LRScheduler &) = delete;
};

} // namespace dense

#endif // OPTIM_LR_SCHEDULER_H_