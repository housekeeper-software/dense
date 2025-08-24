#ifndef LAYER_CNN_H_
#define LAYER_CNN_H_

#include <stdexcept>
#include <stdint.h>

namespace dense {
// 对输入的单个批次填充
// in: 输入数据指针
// channel: 通道数
// in_h,in_w: 输入特征图尺寸
// pad_h, pad_w: 填充尺寸
// out: 输出, 在调用之前必须清零
template <typename T>
void cnn_pad(const T *in, const int64_t in_channel, const int64_t in_h,
             const int64_t in_w, const int64_t pad_h, const int64_t pad_w,
             T *out) {
  if (pad_h < 0 || pad_w < 0) {
    throw std::runtime_error("pad_h/pad_w must be positive.");
  }

  const auto out_h = in_h + 2 * pad_h;
  const auto out_w = in_w + 2 * pad_w;

  for (int64_t c = 0; c < in_channel; ++c) {
    auto in_bt = in + c * in_h * in_w;
    auto out_bt = out + c * out_h * out_w;

    // 复制特征图到指定位置
    for (int64_t h = 0; h < in_h; ++h) {
      auto in_ptr = in_bt + h * in_w;
      auto out_ptr = out_bt + (pad_h + h) * out_w + pad_w;
      std::copy_n(in_ptr, in_w, out_ptr);
    }
  }
}

// 将输入的 3D 输入张量 (不包含批次) 展开为 2D
// 张量，用于将卷积运算转化作矩阵乘法 in：输入数据指针,
// 这里输入必须是填充之后的张量 shape: [C, H, W] stride_h,stride_w: 卷积步长
// kernel_h,kernel_w：卷积核尺寸
// out：输出数据指针
template <typename T>
void cnn_im2col(const T *in, const int64_t channels, const int64_t in_h,
                const int64_t in_w, const int64_t stride_h,
                const int64_t stride_w, const int64_t kernel_h,
                const int64_t kernel_w, T *out) {
  if (stride_h <= 0 || stride_w <= 0) {
    throw std::invalid_argument("stride must be positive");
  }
  if (kernel_h <= 0 || kernel_w <= 0) {
    throw std::invalid_argument("kernel size must be positive");
  }

  // 确保输出尺寸为正
  if (in_h < kernel_h || in_w < kernel_w) {
    throw std::invalid_argument("kernel size larger than input");
  }

  // 计算卷积输出特征图形状
  const auto out_h = (in_h - kernel_h) / stride_h + 1;
  const auto out_w = (in_w - kernel_w) / stride_w + 1;

  // 计算展平矩阵形状
  // 一个卷积核一般包含 1 个通道 或者 3 个通道
  // col_h = 卷积输出的大小 (out_h * out_w)
  // col_w = 卷积核的大小乘以卷积核通道数 (channels * kernel_h * kernel_w)
  const auto col_h = out_h * out_w;
  const auto col_w = channels * kernel_h * kernel_w;

  // 这里模拟卷积过程，但不会与卷积核内积，只是将卷积窗口中的输入元素展平为行向量
  // 在相同的卷积窗口位置，对每个通道分别进行卷积展开，然后将展平的向量拼接在一起,
  // 形成 [channels * kernel_h * kernel_w] 的行向量
  for (int64_t h = 0; h < out_h; ++h) {
    for (int64_t w = 0; w < out_w; ++w) {

      for (int64_t c = 0; c < channels; ++c) {
        // 计算本次卷积展开的首地址
        auto in_ptr = in + c * in_h * in_w + h * stride_h * in_w + w * stride_w;
        // (h * out_w + w) 是当前数据块在输出矩阵中的“行索引”,
        // 滑动的次数就是行索引 col_w 是输出矩阵的宽度（总列数） c * kernel_h *
        // kernel_w 是当前通道在该行内的起始“列偏移”
        auto out_ptr =
            out + (h * out_w + w) * col_w + c * (kernel_h * kernel_w);
        // 展平一个卷积窗口
        unfold(in_ptr, in_w, kernel_h, kernel_w, out_ptr);
      }
    }
  }
}

// col2im: 将矩阵重新折叠成特征图
// col: 输入矩阵，形状为 [spatial_out, kernel_size * channels]
// col_shape: col的形状 [spatial_out, kernel_size * channels]
// kernel_h, kernel_w: 卷积核尺寸
// stride_h, stride_w: 前向传播时卷积步长
// pad_h, pad_w: 前向传播时的 pad 尺寸
// out: 输出特征图
template <typename T>
void cnn_col2im(const T *col, const int64_t channels, const int64_t kernel_h,
                const int64_t kernel_w, const int64_t stride_h,
                const int64_t stride_w, const int64_t pad_h,
                const int64_t pad_w, T *out, const int64_t out_h,
                const int64_t out_w) {
  // 计算填充后的尺寸
  const auto padded_h = out_h + 2 * pad_h;
  const auto padded_w = out_w + 2 * pad_w;

  // 计算卷积窗口的数量
  const auto window_h = (padded_h - kernel_h) / stride_h + 1;
  const auto window_w = (padded_w - kernel_w) / stride_w + 1;
  const auto spatial_size = window_h * window_w;

  // col矩阵的列数应该等于 kernel_size * C
  const auto kernel_size = kernel_h * kernel_w;

  // 对每个空间位置
  for (int64_t spatial_idx = 0; spatial_idx < spatial_size; ++spatial_idx) {
    // 计算当前窗口在特征图中的位置
    auto h_idx = spatial_idx / window_w;
    auto w_idx = spatial_idx % window_w;

    // 计算在填充特征图中的实际位置
    auto h_start = h_idx * stride_h;
    auto w_start = w_idx * stride_w;

    // 对每个通道
    for (int64_t c = 0; c < channels; ++c) {
      // 对卷积核中的每个位置
      for (int64_t kh = 0; kh < kernel_h; ++kh) {
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
          // 计算在填充特征图中的位置
          auto h_pos = h_start + kh;
          auto w_pos = w_start + kw;

          // 检查是否在有效区域内（去除填充）
          auto h_orig = h_pos - pad_h;
          auto w_orig = w_pos - pad_w;

          if (h_orig >= 0 && h_orig < out_h && w_orig >= 0 && w_orig < out_w) {
            // 计算col矩阵中的索引
            auto col_idx = spatial_idx * (kernel_size * channels) +
                           c * kernel_size + kh * kernel_w + kw;

            // 计算输出特征图中的索引
            auto out_idx = c * out_h * out_w + h_orig * out_w + w_orig;

            // 累加到输出特征图（因为可能有重叠）
            out[out_idx] += col[col_idx];
          }
        }
      }
    }
  }
}
} // namespace dense
#endif // LAYER_CNN_H_