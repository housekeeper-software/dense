#include "layer/pooling.h"
#include <iostream>
#include <numeric>
#include <random>

namespace dense {

Pooling::Pooling(Context *ctx, const std::string &name, int pool_method,
                 int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                 int64_t stride_w, int64_t pad_h, int64_t pad_w)
    : Layer(ctx, name), method_(pool_method), kernel_h_(kernel_h),
      kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w) {}

dense::Tensor Pooling::forward(const dense::Tensor &input) {
  auto N = input.size(0);
  auto C = input.size(1);
  auto H = input.size(2);
  auto W = input.size(3);

  input_shape_ = input.sizes();

  auto pooled_height = (H + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
  auto pooled_width = (W + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

  if (pad_h_ || pad_w_) {
    if ((pooled_height - 1) * stride_h_ >= H + pad_h_) {
      --pooled_height;
    }
    if ((pooled_width - 1) * stride_w_ >= W + pad_w_) {
      --pooled_width;
    }
  }

  std::vector<int64_t> output_shape = {N, C, pooled_height, pooled_width};

  auto output = dense::Tensor::zeros(input.dtype(), output_shape);
  auto in_ptr = input.const_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();

  const auto spatial_size = pooled_height * pooled_width;

  if (method_ == kMax) {
    if (!max_idx_.is_defined() || max_idx_.sizes() != output_shape) {
      max_idx_ = dense::Tensor::zeros(DType::kInt32, output_shape);
    }
    auto max_idx_ptr = max_idx_.mutable_data_as<int32_t>();
    // max_idx 全部设置为 -1
    std::fill_n(max_idx_ptr, max_idx_.numel(), -1);
    // output 全部设置为 -FLT_MAX
    std::fill_n(out_ptr, output.numel(), -FLT_MAX);

    for (int64_t b = 0; b < N * C; ++b) {
      // 定位输入和输出指针到当前特征图首地址
      auto in_bt = in_ptr + b * H * W;
      auto out_bt = out_ptr + b * spatial_size;
      auto mask_bt = max_idx_ptr + b * spatial_size;

      for (int64_t ph = 0; ph < pooled_height; ++ph) {
        for (int64_t pw = 0; pw < pooled_width; ++pw) {

          // 这里计算准确的滑动边界，需要忽略填充的部分，因为填充的部分不参与计算
          auto hstart = ph * stride_h_ - pad_h_;
          auto wstart = pw * stride_w_ - pad_w_;
          auto hend = (std::min)(hstart + kernel_h_, H);
          auto wend = (std::min)(wstart + kernel_w_, W);

          hstart = (std::max)(hstart, static_cast<int64_t>(0));
          wstart = (std::max)(wstart, static_cast<int64_t>(0));

          // 当前输出特征图的索引
          const auto pool_index = ph * pooled_width + pw;

          for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
              // 当前输入特征图的索引
              const auto index = h * W + w;

              if (in_bt[index] > out_bt[pool_index]) {
                out_bt[pool_index] = in_bt[index];
                mask_bt[pool_index] = static_cast<int>(index);
              }
            }
          }
        }
      }
    }
  } else if (method_ == kAvg) {
    for (int64_t b = 0; b < N * C; ++b) {
      auto in_bt = in_ptr + b * H * W;
      auto out_bt = out_ptr + b * spatial_size;

      for (int64_t ph = 0; ph < pooled_height; ++ph) {
        for (int64_t pw = 0; pw < pooled_width; ++pw) {

          // 这里计算准确的滑动边界，需要忽略填充的部分，因为填充的部分不参与计算
          /*
          将填充的零值考虑进去并不是指把它们加起来，而是将它们计入分母。这种方法在一些深度学习框架中被称为
          include_pad，即在计算平均值时包含填充区域。这样做的好处是，可以保持平均池化对边界的处理方式与池化核在图像内部时的处理方式一致。
          如果没有填充，pool_size 就等于池化核的大小。
          所以，这里计算逻辑与 max_pooling 不同，因为 max_pooling
          只关心有效区域中的最大值
          */
          auto hstart = ph * stride_h_ - pad_h_;
          auto wstart = pw * stride_w_ - pad_w_;
          auto hend = (std::min)(hstart + kernel_h_, H + pad_h_);
          auto wend = (std::min)(wstart + kernel_w_, W + pad_w_);
          // 计算 pool_size，需要包括填充区域
          auto pool_size = (hend - hstart) * (wend - wstart);

          hstart = (std::max)(hstart, static_cast<int64_t>(0));
          wstart = (std::max)(wstart, static_cast<int64_t>(0));
          // 累加的时候又不需要包括填充区域，因为都是零
          hend = (std::min)(hend, H);
          wend = (std::min)(wend, W);

          for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
              out_bt[ph * pooled_width + pw] += in_bt[h * W + w];
            }
          }
          out_bt[ph * pooled_width + pw] /= pool_size;
        }
      }
    }
  } else {
    throw std::runtime_error("Unknown pooling method.");
  }
  return output;
}

dense::Tensor Pooling::backward(const dense::Tensor &grad_output) {
  const auto N = grad_output.size(0);
  const auto C = grad_output.size(1);
  const auto pooled_height = grad_output.size(2);
  const auto pooled_width = grad_output.size(3);

  const auto H_in = input_shape_[2];
  const auto W_in = input_shape_[3];

  auto grad_input = dense::Tensor::zeros(grad_output.dtype(), input_shape_);

  auto grad_out_ptr = grad_output.const_data_as<float>();
  auto grad_in_ptr = grad_input.mutable_data_as<float>();

  const auto spatial_size = pooled_height * pooled_width;
  if (method_ == kMax) {
    auto mask_ptr = max_idx_.const_data_as<int32_t>();

    for (int64_t b = 0; b < N * C; ++b) {
      auto grad_out_bt = grad_out_ptr + b * spatial_size;
      auto grad_in_bt = grad_in_ptr + b * H_in * W_in;
      auto mask_bt = mask_ptr + b * spatial_size;

      for (int64_t ph = 0; ph < pooled_height; ++ph) {
        for (int64_t pw = 0; pw < pooled_width; ++pw) {
          const auto index = ph * pooled_width + pw;
          const auto input_index = mask_bt[index];
          grad_in_bt[input_index] += grad_out_bt[index];
        }
      }
    }
  } else if (method_ == kAvg) {
    for (int64_t b = 0; b < N * C; ++b) {
      auto grad_out_bt = grad_out_ptr + b * spatial_size;
      auto grad_in_bt = grad_in_ptr + b * H_in * W_in;

      for (int64_t ph = 0; ph < pooled_height; ++ph) {
        for (int64_t pw = 0; pw < pooled_width; ++pw) {

          auto hstart = ph * stride_h_ - pad_h_;
          auto wstart = pw * stride_w_ - pad_w_;
          auto hend = (std::min)(hstart + kernel_h_, H_in + pad_h_);
          auto wend = (std::min)(wstart + kernel_w_, W_in + pad_w_);
          auto pool_size = (hend - hstart) * (wend - wstart);
          hstart = (std::max)(hstart, static_cast<int64_t>(0));
          wstart = (std::max)(wstart, static_cast<int64_t>(0));
          hend = (std::min)(hend, H_in);
          wend = (std::min)(wend, W_in);

          auto pool_index = ph * pooled_width + pw;

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              grad_in_bt[h * W_in + w] += grad_out_bt[pool_index] / pool_size;
            }
          }
        }
      }
    }
  } else {
    throw std::runtime_error("Unknown pooling method.");
  }

  return grad_input;
}
} // namespace dense