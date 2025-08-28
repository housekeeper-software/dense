#include "training.h"
#include "base/data_loader.h"
#include "loss/loss.h"
#include "math/vec_math.h"
#include "model.h"
#include "optim/lr_scheduler.h"
#include "optim/optimizer.h"
#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp> // Include OpenCV headers
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

float random_float(float min, float max, std::mt19937 &gen) {
  std::uniform_real_distribution<> dis(min, max);
  return dis(gen);
}

cv::Mat elastic_transform(const cv::Mat &image, float alpha, float sigma,
                          std::mt19937 &rng) {
  cv::Mat map_x(image.size(), CV_32FC1);
  cv::Mat map_y(image.size(), CV_32FC1);

  // 创建随机位移场
  cv::Mat dx(image.size(), CV_32FC1);
  cv::Mat dy(image.size(), CV_32FC1);
  // 使用 std::normal_distribution 生成高斯噪声
  std::normal_distribution<float> normal_dist(0.0, 1.0); // 均值0，标准差1

  for (int r = 0; r < image.rows; ++r) {
    for (int c = 0; c < image.cols; ++c) {
      dx.at<float>(r, c) = normal_dist(rng);
      dy.at<float>(r, c) = normal_dist(rng);
    }
  }

  // 应用高斯滤波平滑位移场
  int kernel_size = static_cast<int>(sigma * 3.0); // 确保核大小与sigma匹配
  if (kernel_size % 2 == 0)
    kernel_size++; // 确保是奇数
  if (kernel_size < 3)
    kernel_size = 3; // 最小核大小

  cv::GaussianBlur(dx, dx, cv::Size(kernel_size, kernel_size), sigma);
  cv::GaussianBlur(dy, dy, cv::Size(kernel_size, kernel_size), sigma);

  // 根据 alpha 缩放位移场，并创建映射
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      map_x.at<float>(y, x) = x + dx.at<float>(y, x) * alpha;
      map_y.at<float>(y, x) = y + dy.at<float>(y, x) * alpha;
    }
  }

  cv::Mat distorted_image;
  // 使用 cv::remap 应用弹性形变
  // INTER_LINEAR 是线性插值。BORDER_CONSTANT 指定边界模式，Scalar(0.0)
  // 填充黑色。
  cv::remap(image, distorted_image, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar(0.0));

  return distorted_image;
}

dense::Tensor apply_image_augmentations_cnn(const dense::Tensor &input_batch) {
  dense::Tensor augmented_batch = input_batch.clone();
  const auto N = augmented_batch.size(0);
  const auto C = augmented_batch.size(1); // C
  const auto H = augmented_batch.size(2); // H
  const auto W = augmented_batch.size(3); // W

  std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
  // 用于随机决策是否执行某种增强
  std::uniform_real_distribution<> prob_dis(0.0, 1.0);

  // 因为我们只有一个通道，所以把问题简化一下，不用考虑从 [C,H,W]-> [H,W,C]
  for (int64_t i = 0; i < N; ++i) {
    auto image_ptr = augmented_batch.mutable_data_as<float>() + i * C * H * W;
    cv::Mat image_mat(H, W, CV_32FC1, image_ptr);

    // 随机亮度调整
    if (prob_dis(gen) > 0.5) {
      float delta_brightness = random_float(-0.2, 0.2, gen); // 更大的调整范围
      image_mat += delta_brightness;
      cv::max(image_mat, 0.0, image_mat); // 裁剪到 [0, 1] 范围
      cv::min(image_mat, 1.0, image_mat);
    }

    // 随机对比度调整
    if (prob_dis(gen) > 0.5) {
      float alpha_contrast =
          random_float(0.8, 1.2, gen); // 对比度因子 [0.8, 1.2]
      image_mat *= alpha_contrast;
      cv::max(image_mat, 0.0, image_mat); // 裁剪到 [0, 1] 范围
      cv::min(image_mat, 1.0, image_mat);
    }

    // 随机高斯模糊 (小核，避免过度模糊)
    if (prob_dis(gen) > 0.4) { // 较低的概率
      int kernel_size_val =
          static_cast<int>(random_float(0, 1, gen) > 0.5 ? 3 : 5);
      if (kernel_size_val % 2 == 0) { // 确保 kernel_size 是奇数
        kernel_size_val++;
      }
      cv::GaussianBlur(image_mat, image_mat,
                       cv::Size(kernel_size_val, kernel_size_val), 0);
    }

    // 随机旋转 (使用 warpAffine)
    if (prob_dis(gen) > 0.6) {                      // 例如，60% 的概率进行旋转
      float angle = random_float(-15.0, 15.0, gen); // 旋转角度范围 -15 到 15 度
      float scale = 1.0; // 初始缩放为 1.0，可以进一步随机化

      cv::Point2f center((W - 1) / 2.0F, (H - 1) / 2.0F);
      cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
      cv::Size output_size(W, H);

      cv::warpAffine(
          image_mat, image_mat, rot_mat, output_size, cv::INTER_LINEAR,
          cv::BORDER_CONSTANT,
          cv::Scalar(0, 0, 0)); // 灰度图Scalar(0)，彩色图Scalar(0,0,0)
    }

    // 随机平移 (Translation)
    if (prob_dis(gen) > 0.6) { // 60% 概率进行平移
      // 最大平移像素数，可以根据图像大小调整。例如，图像高度的 10%
      float max_translation_pixels_h = H * 0.15;
      float max_translation_pixels_w = W * 0.15;

      float tx = random_float(-max_translation_pixels_w,
                              max_translation_pixels_w, gen);
      float ty = random_float(-max_translation_pixels_h,
                              max_translation_pixels_h, gen);

      cv::Mat trans_mat = (cv::Mat_<float>(2, 3) << 1, 0, tx, 0, 1, ty);
      cv::warpAffine(image_mat, image_mat, trans_mat, image_mat.size(),
                     cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                     cv::Scalar(0, 0, 0)); // 填充黑色
    }

    // 随机缩放 (Scaling)
    if (prob_dis(gen) > 0.6) {                          // 60% 概率进行缩放
      float scale_factor = random_float(0.8, 1.2, gen); // 缩放范围 [0.8, 1.2]

      cv::Point2f center((W - 1) / 2.0F, (H - 1) / 2.0F);
      // getRotationMatrix2D 也可以用于缩放 (角度设为 0)
      cv::Mat scale_mat = cv::getRotationMatrix2D(center, 0, scale_factor);
      cv::warpAffine(image_mat, image_mat, scale_mat, image_mat.size(),
                     cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                     cv::Scalar(0, 0, 0)); // 填充黑色
    }

    //  弹性形变 (Elastic Distortion)
    if (prob_dis(gen) > 0.5) { // 例如，20% 的概率应用弹性形变
      float alpha = random_float(30.0, 40.0, gen); // 形变强度
      float sigma = random_float(5.0, 7.0, gen);   // 高斯核标准差
      // 弹性形变函数应能处理单通道或多通道图像
      image_mat = elastic_transform(image_mat, alpha, sigma, gen);
    }
  }
  // clamp(0.0,1.0)
  auto ptr = augmented_batch.mutable_data_as<float>();
  for (size_t i = 0; i < augmented_batch.numel(); ++i) {
    if (ptr[i] < 0.0f) {
      ptr[i] = 0.0f;
    } else if (ptr[i] > 1.0f) {
      ptr[i] = 1.0f;
    }
  }
  return augmented_batch;
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

      // auto data_augmentations = apply_image_augmentations_cnn(example.data);
      auto data_augmentations = example.data;
      auto logits = model->forward(data_augmentations);
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