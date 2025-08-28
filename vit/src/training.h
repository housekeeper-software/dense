#ifndef TRAINING_H_
#define TRAINING_H_

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace dense {
class DataLoader;
class Loss;
class LRScheduler;
class Optimizer;
} // namespace dense

class VitModel;

// 定义早停的默认参数
const int kDefaultPatience =
    10; // 默认在验证准确率连续10个eval_interval没有改善后停止
const double kDefaultMinDelta = 1e-4; // 认为验证准确率有改善的最小阈值

struct TrainingArguments {
  int epochs;
  int accumulation_steps;
  std::optional<float> max_grad_norm;
  int eval_interval;
  int patience;
  double min_delta;

  TrainingArguments()
      : epochs(0), accumulation_steps(1), max_grad_norm(std::nullopt),
        eval_interval(1), patience(kDefaultPatience),
        min_delta(kDefaultMinDelta) {}
};

std::map<std::string, std::vector<double>>
train(VitModel *model, const std::string &model_dir,
      dense::DataLoader *train_data_loader, dense::DataLoader *test_data_loader,
      dense::Loss *loss_func, dense::Optimizer *optimizer,
      dense::LRScheduler *scheduler, const TrainingArguments &args);
#endif // TRAINING_H_