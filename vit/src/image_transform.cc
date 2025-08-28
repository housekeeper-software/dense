#include "image_transform.h"
#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp> // Include OpenCV headers
#include <random>

namespace {

float random_float(float min, float max, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dis(min, max);
  return dis(gen);
}

// 弹性形变函数
cv::Mat elastic_transform(const cv::Mat &image, float alpha, float sigma,
                          std::mt19937 &rng) {
  cv::Mat map_x(image.size(), CV_32FC1);
  cv::Mat map_y(image.size(), CV_32FC1);

  // 创建随机位移场
  cv::Mat dx(image.size(), CV_32FC1);
  cv::Mat dy(image.size(), CV_32FC1);
  std::normal_distribution<float> normal_dist(0.0, 1.0);

  for (int r = 0; r < image.rows; ++r) {
    for (int c = 0; c < image.cols; ++c) {
      dx.at<float>(r, c) = normal_dist(rng);
      dy.at<float>(r, c) = normal_dist(rng);
    }
  }

  // 应用高斯滤波平滑位移场
  int kernel_size = static_cast<int>(sigma * 3.0);
  if (kernel_size % 2 == 0)
    kernel_size++;
  if (kernel_size < 3)
    kernel_size = 3;

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
  cv::remap(image, distorted_image, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar(0.0));

  return distorted_image;
}

// 应用随机仿射变换（平移、旋转、缩放）
void apply_affine_transform(cv::Mat &image, int H, int W, std::mt19937 &gen) {
  float angle = random_float(-15.0, 15.0, gen);
  float scale = random_float(0.9, 1.1, gen);       // 缩放范围更小，避免失真
  float tx = random_float(-W * 0.1, W * 0.1, gen); // 平移范围更小
  float ty = random_float(-H * 0.1, H * 0.1, gen);

  cv::Point2f center((W - 1) / 2.0f, (H - 1) / 2.0f);
  cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);

  // 在旋转矩阵中加入平移量
  rot_mat.at<double>(0, 2) += tx;
  rot_mat.at<double>(1, 2) += ty;

  cv::warpAffine(image, image, rot_mat, image.size(), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar(0.0));
}

// 应用随机擦除
void apply_random_erasing(cv::Mat &image, std::mt19937 &gen) {
  float prob = 0.5; // 50% 概率执行擦除
  if (random_float(0.0, 1.0, gen) > prob) {
    return;
  }

  int H = image.rows;
  int W = image.cols;

  // 擦除区域的面积比例
  float area_ratio = random_float(0.02, 0.2, gen); // 面积占比 2% 到 20%
  int erase_area = static_cast<int>(H * W * area_ratio);

  // 宽高比
  float aspect_ratio = random_float(0.3, 3.3, gen); // 宽高比范围
  int h_erase = static_cast<int>(std::sqrt(erase_area * aspect_ratio));
  int w_erase = static_cast<int>(std::sqrt(erase_area / aspect_ratio));

  if (h_erase >= H || w_erase >= W) {
    return;
  }

  int x = random_float(0, W - w_erase, gen);
  int y = random_float(0, H - h_erase, gen);

  cv::Rect roi(x, y, w_erase, h_erase);
  cv::Mat sub_image = image(roi);
  sub_image.setTo(cv::Scalar(0.0)); // 擦除区域填充为黑色
}
} // namespace

dense::Tensor apply_image_augmentations_vit(const dense::Tensor &input_batch) {
  dense::Tensor augmented_batch = input_batch.clone();
  const auto N = augmented_batch.size(0);
  const auto C = augmented_batch.size(1);
  const auto H = augmented_batch.size(2);
  const auto W = augmented_batch.size(3);

  std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<> prob_dis(0.0, 1.0);

  for (int64_t i = 0; i < N; ++i) {
    // 因为我们只有一个通道，所以把问题简化一下，不用考虑从 [C,H,W]-> [H,W,C]
    auto image_ptr = augmented_batch.mutable_data_as<float>() + i * C * H * W;
    cv::Mat image_mat(H, W, CV_32FC1, image_ptr);

    // 弹性形变：专门为手写数字设计，非常关键
    if (prob_dis(gen) < 0.5) {
      float alpha = random_float(30.0, 40.0, gen);
      float sigma = random_float(4.0, 6.0, gen); // sigma 范围更小
      image_mat = elastic_transform(image_mat, alpha, sigma, gen);
    }

    // 仿射变换：组合了旋转、平移和缩放
    if (prob_dis(gen) < 0.7) {
      apply_affine_transform(image_mat, H, W, gen);
    }

    // 随机擦除：提升模型对部分遮挡的鲁棒性
    apply_random_erasing(image_mat, gen);

    // 像素级增强：随机调整亮度和对比度
    if (prob_dis(gen) < 0.5) {
      float delta_brightness = random_float(-0.1, 0.1, gen);
      image_mat += delta_brightness;
    }

    if (prob_dis(gen) < 0.5) {
      float alpha_contrast = random_float(0.9, 1.1, gen);
      image_mat *= alpha_contrast;
    }

    // 高斯模糊：降低模型对微小噪音的敏感度
    if (prob_dis(gen) < 0.3) { // 较低的概率
      int kernel_size_val = (prob_dis(gen) < 0.5) ? 3 : 5;
      cv::GaussianBlur(image_mat, image_mat,
                       cv::Size(kernel_size_val, kernel_size_val), 0);
    }
  }

  // 统一将所有像素值裁剪到 [0.0, 1.0] 范围内
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
