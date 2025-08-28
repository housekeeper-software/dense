#include "training.h"
#include "base/data_loader.h"
#include "image_transform.h"
#include "loss/loss.h"
#include "math/vec_math.h"
#include "model.h"
#include "optim/lr_scheduler.h"
#include "optim/optimizer.h"
#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <random>

namespace {
void WriteLog(const std::string &dir, const std::string &str) {
  std::string filename = dir + "/train.log";
  FILE *fp = fopen(filename.c_str(), "a");
  fprintf(fp, "%s\n", str.c_str());
  fclose(fp);
}

std::pair<double, double> evaluate(VitModel *model,
                                   dense::DataLoader *test_data_loader,
                                   dense::Loss *loss_func) {
  double total_loss = 0.0;
  int64_t num_batches = 0;
  int64_t total_accuracy = 0;

  for (const auto &batch : *test_data_loader) {
    if (batch.empty())
      break;
    auto example = dense::DataLoader::apply_batch(batch);
    auto logits = model->forward(example.data);
    auto loss = loss_func->forward(logits, example.target);
    total_loss += loss;
    num_batches++;

    // 计算正确率
    auto N = logits.size(0);
    auto C = logits.size(1);

    vec::mat_softmax_forward_blas(logits.mutable_data_as<float>(), N, C);

    auto target_ptr = example.target.const_data_as<int64_t>();

    for (int64_t n = 0; n < N; ++n) {
      auto logits_bt = logits.const_data_as<float>() + n * C;
      float max_value = -INFINITY;
      int32_t max_idx = -1;
      for (int64_t c = 0; c < C; ++c) {
        if (logits_bt[c] > max_value) {
          max_value = logits_bt[c];
          max_idx = c;
        }
      }
      if (max_idx == target_ptr[n]) {
        ++total_accuracy;
      }
    }
  }
  if (num_batches < 1)
    return {0.0f, 0.0f};
  return {static_cast<double>(total_loss) / static_cast<double>(num_batches),
          static_cast<double>(total_accuracy) /
              static_cast<double>(test_data_loader->dataset_size())};
}
} // namespace

std::map<std::string, std::vector<double>>
train(VitModel *model, const std::string &model_dir,
      dense::DataLoader *train_data_loader, dense::DataLoader *test_data_loader,
      dense::Loss *loss_func, dense::Optimizer *optimizer,
      dense::LRScheduler *scheduler, const TrainingArguments &args) {
  // 1. 初始化
  std::map<std::string, std::vector<double>> history;
  double best_val_loss = std::numeric_limits<double>::max();
  int patience_counter = 0;

  model->enable_training(true); // 确保模型处于训练模式

  double total_train_batches = static_cast<double>(train_data_loader->size());

  // 2. 主训练循环 (按 Epoch)
  for (int epoch = 0; epoch < args.epochs; ++epoch) {
    // --- 训练阶段 ---
    double running_loss = 0.0;
    int batch_count = 0;
    model->clear_grads();
    double accumulated_loss = 0.0;

    // 遍历训练数据加载器中的所有批次
    for (auto &batch : *train_data_loader) {
      if (batch.empty())
        break;

      auto example = dense::DataLoader::apply_batch(batch);
      auto input = example.data;
      if (args.use_image_transform) {
        input = apply_image_augmentations_vit(example.data);
      }
      auto logits = model->forward(input);
      auto loss = loss_func->forward(logits, example.target);
      auto grad_loss = loss_func->backward();

      if (args.accumulation_steps > 1) {
        // 在进行反向传播之前，将梯度除以累积步数
        // 这样可以确保在累积梯度后，更新的幅度与一个大批次更新的幅度大致相同
        auto N = grad_loss.numel();
        auto ptr = grad_loss.mutable_data_as<float>();
        for (size_t i = 0; i < N; ++i) {
          ptr[i] = ptr[i] / args.accumulation_steps;
        }
      }
      model->backward(grad_loss);

      accumulated_loss += loss;
      batch_count++;

      double current_epoch_progress =
          static_cast<double>(epoch) +
          (static_cast<double>(batch_count) / total_train_batches);

      scheduler->step(current_epoch_progress);
      auto lr = scheduler->get_lr();
      optimizer->set_lr(lr);

      {
        char buf[1024] = {0};
        sprintf(buf, "Epoch:%d, Batch:%d,Loss: %.8f,LR: %.8f", epoch + 1,
                batch_count, loss, lr);
        WriteLog(model_dir, buf);
      }

      std::cout << "Epoch:" << epoch + 1 << ",Batch:" << batch_count
                << ", Training Loss: " << loss << ", LR: " << std::fixed
                << std::setprecision(8) << lr << std::endl;

      if (batch_count % args.accumulation_steps == 0) {

        std::vector<dense::ParamsAndGrads> params_and_grads;
        model->get_params_and_grads(params_and_grads);

        // 使用优化器更新参数
        optimizer->update(params_and_grads, args.max_grad_norm);
        model->clear_grads();

        running_loss += accumulated_loss / args.accumulation_steps;

        {
          char buf[1024] = {0};
          sprintf(buf, "Epoch:%d, Batch:%d,Loss (accumulated): %.8f", epoch + 1,
                  batch_count, accumulated_loss / args.accumulation_steps);
          WriteLog(model_dir, buf);
        }

        std::cout << "Epoch:" << epoch + 1 << ",Batch:" << batch_count
                  << ", Training Loss (accumulated): "
                  << accumulated_loss / args.accumulation_steps << std::endl;

        // 重置累积损失
        accumulated_loss = 0.0;
      }
    }

    // 处理最后一个不完整的梯度累积批次
    if (batch_count % args.accumulation_steps != 0) {
      std::vector<dense::ParamsAndGrads> params_and_grads;
      model->get_params_and_grads(params_and_grads);

      optimizer->update(params_and_grads, args.max_grad_norm);
      model->clear_grads();
      running_loss +=
          accumulated_loss / (batch_count % args.accumulation_steps);

      {
        char buf[1024] = {0};
        sprintf(buf, "Epoch:%d, Batch:%d,Loss (accumulated): %.8f", epoch + 1,
                batch_count, accumulated_loss / args.accumulation_steps);
        WriteLog(model_dir, buf);
      }

      std::cout << "Epoch:" << epoch + 1 << ",Batch:" << batch_count
                << ", Training Loss (accumulated): "
                << accumulated_loss / args.accumulation_steps << std::endl;
    }

    double avg_train_loss = running_loss / batch_count;
    history["train_loss"].push_back(avg_train_loss);
    std::cout << "Epoch " << epoch + 1
              << " Average Train Loss: " << avg_train_loss << std::endl;

    // --- 评估阶段 ---
    if ((epoch + 1) % args.eval_interval == 0) {
      model->enable_training(false); // 禁用训练模式
      auto val_result = evaluate(model, test_data_loader, loss_func);

      model->enable_training(true); // 恢复训练模式
      history["val_loss"].push_back(val_result.first);
      std::cout << "Epoch " << epoch + 1
                << " Validation Loss: " << val_result.first
                << ", Validation accuracy: " << val_result.second << std::endl;

      // --- 早停与保存最佳模型逻辑 ---
      if (val_result.first < best_val_loss - args.min_delta) {
        best_val_loss = val_result.first;
        patience_counter = 0;
        char buffer[1024];
        snprintf(buffer, sizeof(buffer), "%s/best_%d_%.4f.safetensors",
                 model_dir.c_str(), epoch + 1, val_result.second);
        std::cout << "Validation loss improved. Saving model to " << buffer
                  << std::endl;
        model->save(buffer);
      } else {
        patience_counter++;
        std::cout << "Validation loss did not improve. Patience: "
                  << patience_counter << "/" << args.patience << std::endl;
      }

      if (patience_counter >= args.patience) {
        std::cout << "Early stopping triggered after " << epoch + 1
                  << " epochs." << std::endl;
        break; // 退出训练循环
      }
    }
  }
  std::cout << "Training finished." << std::endl;
  return history;
}