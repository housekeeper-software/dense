#include "layer/batch_norm.h"
#include "layer/init.h"
#include "math/vec_math.h"
#include <stdexcept>

namespace dense {

namespace {

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
batch_norm_forward_native(bool training, const Tensor &input,
                          const Tensor &weight, const Tensor &bias,
                          const Tensor &running_mean, const Tensor &running_var,
                          float eps, bool affine, bool track_running_stats) {

  const auto N = input.size(0); // 批次
  const auto C = input.size(1); // 特征维度

  // 输入数据的空间维度，也就是每个通道中，单个特征图的像素总数
  // 一般来说是 H * W
  const int64_t spatial_dim = input.numel() / (N * C);
  // mini-batch 总元素 , 对于CNN，通常是 N×H×W，不包含特征通道
  const int64_t batch_sum = N * spatial_dim;

  auto mean = Tensor::zeros(DType::kFloat32, {C}); // 均值
  auto var = Tensor::zeros(DType::kFloat32, {C});  // 方差
  auto rstd = Tensor::zeros(DType::kFloat32, {C}); // 标准差的倒数
  auto x_norm = Tensor::zeros_like(input);         // 归一化后的输入
  auto output = Tensor::zeros_like(input);         // 前向输出

  auto mean_ptr = mean.mutable_data_as<float>();
  auto var_ptr = var.mutable_data_as<float>();
  auto rstd_ptr = rstd.mutable_data_as<float>();
  auto x_norm_ptr = x_norm.mutable_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();

  auto in_ptr = input.const_data_as<float>();
  auto weight_ptr = weight.const_data_as<float>();
  auto bias_ptr = bias.const_data_as<float>();

  if (!training && track_running_stats) {
    // 推理阶段，并且存在 running_mean,running_var(训练所得)
    // 则使用滑动均值和方差来归一化
    auto running_mean_ptr = running_mean.const_data_as<float>();
    auto running_var_ptr = running_var.const_data_as<float>();

    // 计算 rstd = 1 / sqrt(var + eps)
    for (size_t i = 0; i < rstd.numel(); ++i) {
      rstd_ptr[i] = 1.0f / (std::sqrt(running_var_ptr[i] + eps));
    }
    // 归一化 x_hat = (x-mean)/sqrt(var+eps)
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {

        auto in_bt = in_ptr + n * C * spatial_dim + c * spatial_dim;
        auto x_norm_bt = x_norm_ptr + n * C * spatial_dim + c * spatial_dim;
        auto out_bt = out_ptr + n * C * spatial_dim + c * spatial_dim;

        for (int64_t i = 0; i < spatial_dim; ++i) {
          x_norm_bt[i] = (in_bt[i] - running_mean_ptr[c]) * rstd_ptr[c];

          if (affine) {
            // 进行仿射变换：y = gamma * y + beta
            out_bt[i] = x_norm_bt[i] * weight_ptr[c] + bias_ptr[c];
          } else {
            // 不需要仿射变换
            out_bt[i] = x_norm_bt[i];
          }
        }
      }
    }
    // 此刻 mean, var 无效，推理阶段，running_mean,running_var 已经固定
    // 不需要更新
    return std::make_tuple(output, mean, var, rstd, x_norm);
  }

  // 只要 track_running_stats = false，我们就要计算当前批次的均值和方差
  // 因为没有 running_mean,running_var 可以使用，这与 pytorch 是一致的

  // 计算均值 mean = 1/m* sum(x_i)
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      auto in_bt = in_ptr + n * C * spatial_dim + c * spatial_dim;

      for (int64_t i = 0; i < spatial_dim; ++i) {
        mean_ptr[c] += in_bt[i];
      }
    }
  }

  for (size_t i = 0; i < mean.numel(); ++i) {
    mean_ptr[i] /= static_cast<float>(batch_sum);
  }

  // 计算方差 var = 1/m * sum(x_i-mean)^2
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      auto in_bt = in_ptr + n * C * spatial_dim + c * spatial_dim;

      float sum = 0.0f;
      for (int64_t i = 0; i < spatial_dim; ++i) {
        const auto x_mean = in_bt[i] - mean_ptr[c];
        sum += x_mean * x_mean;
      }
      var_ptr[c] += sum;
    }
  }
  for (size_t i = 0; i < var.numel(); ++i) {
    var_ptr[i] /= static_cast<float>(batch_sum);
  }

  // 计算标准差的倒数： rstd = 1 / sqrt(var + eps)
  for (size_t i = 0; i < rstd.numel(); ++i) {
    rstd_ptr[i] = 1.0f / (std::sqrt(var_ptr[i] + eps));
  }

  // 归一化 x_hat = (x-mean)/sqrt(var+eps)
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      auto in_bt = in_ptr + n * C * spatial_dim + c * spatial_dim;
      auto x_norm_bt = x_norm_ptr + n * C * spatial_dim + c * spatial_dim;
      auto out_bt = out_ptr + n * C * spatial_dim + c * spatial_dim;

      for (int64_t i = 0; i < spatial_dim; ++i) {
        x_norm_bt[i] = (in_bt[i] - mean_ptr[c]) * rstd_ptr[c];

        if (affine) {
          // 进行仿射变换：y = gamma * y + beta
          out_bt[i] = x_norm_bt[i] * weight_ptr[c] + bias_ptr[c];
        } else {
          // 不需要仿射变换，直接输出
          out_bt[i] = x_norm_bt[i];
        }
      }
    }
  }
  // 这里所有的值都是有效的
  return std::make_tuple(output, mean, var, rstd, x_norm);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
batch_norm_forward_blas(bool training, const Tensor &input,
                        const Tensor &weight, const Tensor &bias,
                        const Tensor &running_mean, const Tensor &running_var,
                        float eps, bool affine, bool track_running_stats) {
  const auto N = input.size(0); // 批次
  const auto C = input.size(1); // 特征维度

  // 输入数据的空间维度，也就是每个通道中，单个特征图的像素总数
  // 一般来说是 H * W
  const int64_t spatial_dim = input.numel() / (N * C);
  // mini-batch 总元素 , 对于CNN，通常是 N×H×W
  const int64_t batch_sum = N * spatial_dim;

  float batch_sum_scale = 1.0f / static_cast<float>(batch_sum);

  // 生成一个足够长度的全1向量
  std::vector<float> ones(std::max(N * C, spatial_dim), 1.0f);

  auto mean = Tensor::zeros(DType::kFloat32, {C}); // 均值
  auto var = Tensor::zeros(DType::kFloat32, {C});  // 方差
  auto rstd = Tensor::zeros(DType::kFloat32, {C}); // 标准差的倒数
  auto x_norm = Tensor::zeros_like(input);         // 归一化后的输入
  auto output = Tensor::zeros_like(input);         // 前向输出

  auto mean_ptr = mean.mutable_data_as<float>();
  auto var_ptr = var.mutable_data_as<float>();
  auto rstd_ptr = rstd.mutable_data_as<float>();
  auto x_norm_ptr = x_norm.mutable_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();

  auto in_ptr = input.const_data_as<float>();
  auto weight_ptr = weight.const_data_as<float>();
  auto bias_ptr = bias.const_data_as<float>();

  if (!training && track_running_stats) {
    // 推理阶段，并且存在 running_mean,running_var(训练所得)
    // 则使用滑动均值和方差来归一化
    auto running_mean_ptr = running_mean.const_data_as<float>();
    auto running_var_ptr = running_var.const_data_as<float>();

    // 计算 rstd = 1 / sqrt(var + eps)
    // 这个地方不值用 cblas 优化，特征维度很小
    for (size_t i = 0; i < rstd.numel(); ++i) {
      rstd_ptr[i] = 1.0f / (std::sqrt(running_var_ptr[i] + eps));
    }

    // 归一化 x_hat = (x-mean)/sqrt(var+eps)
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {

        auto in_bt = in_ptr + n * C * spatial_dim + c * spatial_dim;
        auto out_bt = out_ptr + n * C * spatial_dim + c * spatial_dim;
        auto x_norm_bt = x_norm_ptr + n * C * spatial_dim + c * spatial_dim;

        // x_norm_bt = in_bt
        vec::scopy_blas(spatial_dim, in_bt, 1, x_norm_bt, 1);
        // x_norm_bt = x_norm_bt - mean
        vec::saxpy_blas(spatial_dim, -running_mean_ptr[c], ones.data(), 1,
                        x_norm_bt, 1);
        // x_norm_bt = x_norm_bt * rstd
        vec::sscal_blas(spatial_dim, rstd_ptr[c], x_norm_bt, 1);
        // out_bt = x_norm_bt
        vec::scopy_blas(spatial_dim, x_norm_bt, 1, out_bt, 1);

        if (affine) {
          // 仿射变换
          // out_bt = out_bt * gamma
          vec::sscal_blas(spatial_dim, weight_ptr[c], out_bt, 1);
          // out_bt = out_bt + beta
          vec::saxpy_blas(spatial_dim, 1.0f, &bias_ptr[c], 0, out_bt, 1);
        }
      }
    }
    return std::make_tuple(output, mean, var, rstd, x_norm);
  }

  std::vector<float> num_by_chans(N * C, 0.0f);
  std::vector<float> temp(spatial_dim, 0.0f);

  // 计算均值 mean = 1/m* sum(x_i)
  // 将输入重塑为 [C*N, H*W], 然后与 全 1 向量乘法，生成一个 [C*N] 的向量
  // 这就相当于每个 H*W 数据与 全(1)向量内积，相当于求和
  vec::sgemv_blas(N * C, spatial_dim, batch_sum_scale, in_ptr, spatial_dim,
                  ones.data(), 1, 0.0f, num_by_chans.data(), 1);

  // 将 num_by_chans [C*N] 重塑为
  // [N,C]，然后用它的转置与全(1)向量内积，等价于在批次维度再次求和，
  // 最终生成一个 [C] 的向量，保存到 mean 中
  vec::sgemv_transpose_blas(N, C, 1.0f, num_by_chans.data(), C, ones.data(), 1,
                            0.0f, mean_ptr, 1);

  // 计算方差 var = 1/m * sum(x_i-mean)^2
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      auto in_bt = in_ptr + n * C * spatial_dim + c * spatial_dim;

      // y = x
      vec::scopy_blas(spatial_dim, in_bt, 1, temp.data(), 1);
      // y = y-mean
      vec::saxpy_blas(spatial_dim, -mean_ptr[c], ones.data(), 1, temp.data(),
                      1);
      // sum(y-mean)^2, 我们用自己和自己内积来实现
      var_ptr[c] += vec::sdot_blas(spatial_dim, temp.data(), 1, temp.data(), 1);
    }
  }
  // var = var/m
  vec::sscal_blas(C, batch_sum_scale, var_ptr, 1);

  // 计算 rstd = 1 / sqrt(var + eps)
  for (size_t i = 0; i < rstd.numel(); ++i) {
    rstd_ptr[i] = 1.0f / (std::sqrt(var_ptr[i] + eps));
  }

  // 归一化 x_hat = (x-mean)/sqrt(var+eps)
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      auto in_bt = in_ptr + n * C * spatial_dim + c * spatial_dim;
      auto x_norm_bt = x_norm_ptr + n * C * spatial_dim + c * spatial_dim;
      auto out_bt = out_ptr + n * C * spatial_dim + c * spatial_dim;

      // 使用 BLAS 计算归一化
      vec::scopy_blas(spatial_dim, in_bt, 1, x_norm_bt, 1);
      vec::saxpy_blas(spatial_dim, -mean_ptr[c], ones.data(), 1, x_norm_bt, 1);
      vec::sscal_blas(spatial_dim, rstd_ptr[c], x_norm_bt, 1);
      vec::scopy_blas(spatial_dim, x_norm_bt, 1, out_bt, 1);

      if (affine) {
        vec::sscal_blas(spatial_dim, weight_ptr[c], out_bt, 1);
        vec::saxpy_blas(spatial_dim, 1.0f, &bias_ptr[c], 0, out_bt, 1);
      }
    }
  }
  return std::make_tuple(output, mean, var, rstd, x_norm);
}

dense::Tensor batch_norm_backward_native(const dense::Tensor &grad_output,
                                         const dense::Tensor &gamma,
                                         const dense::Tensor &x_norm,
                                         const dense::Tensor &mean,
                                         const dense::Tensor &rstd,
                                         dense::Tensor &gamma_grad,
                                         dense::Tensor &beta_grad) {
  const auto N = grad_output.size(0);
  const auto C = grad_output.size(1);
  const int64_t spatial_dim = grad_output.numel() / (N * C);

  auto grad_input = dense::Tensor::zeros_like(grad_output);

  auto gamma_ptr = gamma.const_data_as<float>();
  auto x_norm_ptr = x_norm.const_data_as<float>();
  auto mean_ptr = mean.const_data_as<float>();
  auto rstd_ptr = rstd.const_data_as<float>();

  auto grad_in_ptr = grad_input.mutable_data_as<float>();
  auto grad_out_ptr = grad_output.const_data_as<float>();

  // mini-batch 总元素 , 对于CNN，通常是 N×H×W
  const int64_t batch_sum = N * spatial_dim;

  float batch_sum_scale = 1.0f / static_cast<float>(batch_sum);

  if (gamma_grad.is_defined() && beta_grad.is_defined()) {
    auto gamma_grad_ptr = gamma_grad.mutable_data_as<float>();
    auto beta_grad_ptr = beta_grad.mutable_data_as<float>();
    // $$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m}\frac{\partial
    // L}{\partial y_{i}} \cdot \hat{x}_{i}$$
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        auto grad_out_bt = grad_out_ptr + n * C * spatial_dim + c * spatial_dim;
        auto x_norm_bt = x_norm_ptr + n * C * spatial_dim + c * spatial_dim;
        for (int64_t i = 0; i < spatial_dim; ++i) {
          gamma_grad_ptr[c] += grad_out_bt[i] * x_norm_bt[i];
        }
      }
    }

    // $$\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m}\frac{\partial
    // L}{\partial y_{i}}$$
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < C; ++c) {
        auto grad_out_bt = grad_out_ptr + n * C * spatial_dim + c * spatial_dim;
        for (int64_t i = 0; i < spatial_dim; ++i) {
          beta_grad_ptr[c] += grad_out_bt[i];
        }
      }
    }
  }

  std::vector<float> grad_out_sum_per_channel(C, 0.0f);
  std::vector<float> grad_out_product_x_hat_sum_per_channel(C, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      auto grad_out_bt = grad_out_ptr + n * C * spatial_dim + c * spatial_dim;
      auto x_norm_bt = x_norm_ptr + n * C * spatial_dim + c * spatial_dim;

      for (int64_t i = 0; i < spatial_dim; ++i) {
        grad_out_sum_per_channel[c] += grad_out_bt[i];
        grad_out_product_x_hat_sum_per_channel[c] +=
            grad_out_bt[i] * x_norm_bt[i];
      }
    }
  }

  // dL/dx
  /*
   $$\frac{\partial L}{\partial x_{i}}=\frac{\gamma}{\sqrt{\sigma^2+\epsilon
   }}\left(\frac{\partial L}{\partial
   y_{i}}-\frac{1}{m}\sum_{j=1}^{m}\frac{\partial L}{\partial y_{j} } -
   \hat{x}_{i}\frac{1}{m}\sum_{j=1}^{m}\frac{\partial L}{\partial
   y_{j}}\hat{x}_{j}\right)$$
  */
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      auto grad_out_bt = grad_out_ptr + n * C * spatial_dim + c * spatial_dim;
      auto grad_in_bt = grad_in_ptr + n * C * spatial_dim + c * spatial_dim;
      auto x_norm_bt = x_norm_ptr + n * C * spatial_dim + c * spatial_dim;

      float batch_grad_out_sum = grad_out_sum_per_channel[c];
      float batch_grad_out_product_x_hat_sum =
          grad_out_product_x_hat_sum_per_channel[c];

      float gamma_val = gamma.is_defined() ? gamma_ptr[c] : 1.0f;
      for (int64_t i = 0; i < spatial_dim; ++i) {
        grad_in_bt[i] =
            gamma_val * rstd_ptr[c] *
            (grad_out_bt[i] - batch_sum_scale * batch_grad_out_sum -
             x_norm_bt[i] * batch_sum_scale * batch_grad_out_product_x_hat_sum);
      }
    }
  }
  return grad_input;
}

dense::Tensor batch_norm_backward_blas(const dense::Tensor &grad_output,
                                       const dense::Tensor &gamma,
                                       const dense::Tensor &x_norm,
                                       const dense::Tensor &mean,
                                       const dense::Tensor &rstd,
                                       dense::Tensor &gamma_grad,
                                       dense::Tensor &beta_grad) {
  const auto N = grad_output.size(0);
  const auto C = grad_output.size(1);
  const int64_t spatial_dim = grad_output.numel() / (N * C);

  auto grad_input = dense::Tensor::zeros_like(grad_output);

  auto gamma_ptr = gamma.const_data_as<float>();
  auto x_norm_ptr = x_norm.const_data_as<float>();
  auto mean_ptr = mean.const_data_as<float>();
  auto rstd_ptr = rstd.const_data_as<float>();
  auto gamma_grad_ptr = gamma_grad.mutable_data_as<float>();
  auto beta_grad_ptr = beta_grad.mutable_data_as<float>();
  auto grad_in_ptr = grad_input.mutable_data_as<float>();
  auto grad_out_ptr = grad_output.const_data_as<float>();

  // mini-batch 总元素 , 对于CNN，通常是 N×H×W
  const int64_t batch_sum = N * spatial_dim;

  float batch_sum_scale = 1.0f / static_cast<float>(batch_sum);

  std::vector<float> ones(std::max(N * C, spatial_dim), 1.0f);
  std::vector<float> num_by_chans(N * C, 0.0f);

  std::vector<float> grad_out_sum_per_channel(C, 0.0f);
  std::vector<float> grad_out_product_x_hat_sum_per_channel(C, 0.0f);

  // 计算上游梯度的逐通道累加
  vec::sgemv_blas(C * N, spatial_dim, 1.0f, grad_out_ptr, spatial_dim,
                  ones.data(), 1, 0.0f, &num_by_chans[0], 1);

  // 将 num_by_chans [C*N] 重塑为
  // [N,C]，然后用它的转置与全(1)向量内积，等价于在批次维度再次求和，
  // 最终生成一个 [C] 的向量，保存到 grad_out_sum_per_channel 中
  vec::sgemv_transpose_blas(N, C, 1.0f, &num_by_chans[0], C, ones.data(), 1,
                            0.0f, &grad_out_sum_per_channel[0], 1);

  // 计算上游梯度逐通道与前向传播的归一化值的乘积，然后累加
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      auto grad_out_bt = grad_out_ptr + n * C * spatial_dim + c * spatial_dim;
      auto x_norm_bt = x_norm_ptr + n * C * spatial_dim + c * spatial_dim;
      grad_out_product_x_hat_sum_per_channel[c] +=
          vec::sdot_blas(spatial_dim, grad_out_bt, 1, x_norm_bt, 1);
    }
  }

  if (gamma_grad.is_defined() && beta_grad.is_defined()) {
    // 计算 dL/dbeta
    vec::scopy_blas(C, &grad_out_sum_per_channel[0], 1, beta_grad_ptr, 1);

    // 计算 dL/dgamma
    vec::scopy_blas(C, &grad_out_product_x_hat_sum_per_channel[0], 1,
                    gamma_grad_ptr, 1);
  }

  std::vector<float> temp(spatial_dim, 0.0f);

  // 计算 dL/dx
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      auto grad_out_bt = grad_out_ptr + n * C * spatial_dim + c * spatial_dim;
      auto grad_in_bt = grad_in_ptr + n * C * spatial_dim + c * spatial_dim;
      auto x_norm_bt = x_norm_ptr + n * C * spatial_dim + c * spatial_dim;

      float gamma_val = gamma.is_defined() ? gamma_ptr[c] : 1.0f;

      auto t1 = gamma_val * rstd_ptr[c];
      auto t2 = batch_sum_scale * grad_out_sum_per_channel[c];
      auto t3 = batch_sum_scale * grad_out_product_x_hat_sum_per_channel[c];

      // temp 要清零
      std::fill(temp.begin(), temp.end(), 0.0f);

      // temp = x_norm_bt[i] * t2 + 0
      vec::saxpy_blas(spatial_dim, t3, x_norm_bt, 1, &temp[0], 1);

      // temp = grad_out_bt  - t2 - temp
      for (int64_t i = 0; i < spatial_dim; ++i) {
        temp[i] = grad_out_bt[i] - t2 - temp[i];
      }
      // grad_in = t1 * temp + 0, grad_in 要清空
      vec::saxpy_blas(spatial_dim, t1, &temp[0], 1, grad_in_bt, 1);
    }
  }

  return grad_input;
}

} // namespace

BatchNorm::BatchNorm(Context *ctx, const std::string &name,
                     int64_t num_features, float eps, float momentum,
                     bool affine, bool track_running_stats)
    : Layer(ctx, name), num_features_(num_features), eps_(eps),
      momentum_(momentum), affine_(affine),
      track_running_stats_(track_running_stats) {

  if (affine_ || track_running_stats_) {
    RegisterParam();
  }
}

void BatchNorm::init() {
  if (affine_) {
    W_ = Tensor::empty(DType::kFloat32, {num_features_});
    init::ones_(W_);

    b_ = Tensor::zeros(DType::kFloat32, {num_features_});
  }
  if (track_running_stats_) {
    // 均值为零, 方差为1
    running_mean_ = Tensor::zeros(DType::kFloat32, {num_features_});

    running_var_ = Tensor::empty(DType::kFloat32, {num_features_});
    init::ones_(running_var_);
  }
}

Tensor BatchNorm::forward(const Tensor &input) {
  if (input.dim() < 2) {
    throw std::runtime_error("The expected input should be at least 2D.");
  }
  auto [output, mean, var, rstd, x_norm] = forward_device(input);

  if (!is_training()) {
    return output;
  }

  mean_ = mean;
  rstd_ = rstd;
  x_norm_ = x_norm;

  if (!track_running_stats_) {
    // 如果没有跟踪均值和方差，则直接返回输出
    return output;
  }

  const auto N = input.size(0); // 批次
  const auto C = input.size(1); // 特征维度

  const int64_t spatial_dim = input.numel() / (N * C);
  // 在训练阶段，更新滑动平均的均值和方差
  // running_mean = (1 - momentum) * running_mean + momentum * mean

  auto running_mean_ptr = running_mean_.mutable_data_as<float>();
  auto running_var_ptr = running_var_.mutable_data_as<float>();
  auto mean_ptr = mean.const_data_as<float>();
  auto var_ptr = var.const_data_as<float>();

  float unbiased_var = static_cast<float>(N * spatial_dim) /
                       static_cast<float>(N * spatial_dim - 1);

  for (size_t i = 0; i < mean_.numel(); ++i) {
    running_mean_ptr[i] =
        (1 - momentum_) * running_mean_ptr[i] + momentum_ * mean_ptr[i];

    // 转为 无偏方差，需要乘以一个 m/(m-1)，这里的 m = N * spatial_dim,
    // pytorch是这么做的
    running_var_ptr[i] = (1 - momentum_) * running_var_ptr[i] +
                         momentum_ * var_ptr[i] * unbiased_var;
  }
  return output;
}

Tensor BatchNorm::backward(const Tensor &grad_output) {
  auto N = grad_output.size(0);
  auto C = grad_output.size(1);

  if (affine_) {
    if (!grad_W_.is_defined()) {
      grad_W_ = Tensor::zeros_like(W_);
    }
    if (!grad_b_.is_defined()) {
      grad_b_ = Tensor::zeros_like(b_);
    }
  }
  if (ctx()->device.is_blas()) {
    return batch_norm_backward_blas(grad_output, W_, x_norm_, mean_, rstd_,
                                    grad_W_, grad_b_);
  }
  return batch_norm_backward_native(grad_output, W_, x_norm_, mean_, rstd_,
                                    grad_W_, grad_b_);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
BatchNorm::forward_device(const Tensor &input) {
  if (ctx()->device.is_blas()) {
    return batch_norm_forward_blas(is_training(), input, W_, b_, running_mean_,
                                   running_var_, eps_, affine_,
                                   track_running_stats_);
  }
  return batch_norm_forward_native(is_training(), input, W_, b_, running_mean_,
                                   running_var_, eps_, affine_,
                                   track_running_stats_);
}

} // namespace dense