#include "layer/conv2d.h"
#include "layer/cnn.h"
#include "layer/init.h"
#include "math/vec_math.h"
#include <assert.h>

namespace dense {

namespace {

// 将单个卷积窗口展平
// in: 输入数据指针
// in_stride: 输入的行距
// kernel_h,kernel_w：卷积核的高度和宽度
// out: 输出位置指针
template <typename T>
inline void unfold(const T *in, const int64_t in_stride, const int64_t kernel_h,
                   const int64_t kernel_w, T *out) {
  for (int64_t h = 0; h < kernel_h; ++h) {
    auto in_ptr = in + h * in_stride;
    auto out_ptr = out + h * kernel_w;
    std::copy_n(in_ptr, kernel_w, out_ptr);
  }
}

// 2D 卷积
// in: 输入数据指针，是 cnn_im2col 的输出张量
// in_shape: [col_h, col_w]
// kernel：卷积核展开成的向量
// 卷积核的形状 [out_channel,in_channel,kernel_h,kernel_w]
// bias：偏置向量，形状为 [out_channel]，可选
// out: 输出形状 [out_channel,col_h], col_h: 输出特征图尺寸，所以输出等价于:
// [out_channel,out_h,out_w]
void cnn_conv2d(const Device &device, const float *col, const int64_t col_h,
                const int64_t col_w, const float *kernel,
                const int64_t out_channel, const int64_t in_channel,
                const int64_t kernel_h, const int64_t kernel_w,
                const float *bias, float *out, const int64_t out_h,
                const int64_t out_w) {

  // 输出特征图的尺寸
  const auto spatial_size = out_h * out_w;
  // 卷积核尺寸
  const auto channel_kernel_size = in_channel * kernel_h * kernel_w;

  assert(channel_kernel_size == col_w);

  // 这里 kernel 的内存必须是连续的,
  // 虽然它的形状为 [out_channel,in_channel,kernel_h,kernel_w],
  // 但我们可以将之视为 [out_channel, in_channel * kernel_h * kernel_w]
  // 且 channel_kernel_size == col_w，否则矩阵不能相乘

  // kernel: [out_channel, col_w], col^T: [col_w, col_h] -> [out_channel, col_h]
  if (device.is_blas()) {
    vec::matmul_B_transpose_blas(kernel, channel_kernel_size, col, col_w,
                                 nullptr, out, spatial_size, out_channel /*M*/,
                                 col_h /*N*/, col_w /*K*/);
  } else {
    vec::matmul_B_transpose_native(kernel, channel_kernel_size, col, col_w,
                                   nullptr, out, spatial_size,
                                   out_channel /*M*/, col_h /*N*/, col_w /*K*/);
  }
  if (bias) {
    for (int64_t c = 0; c < out_channel; ++c) {
      auto out_ptr = out + c * spatial_size;
      // 每个卷积输出的所有元素都要加上当前通道的偏置
      const float bias_val = bias[c];

      //  Y = Y + alpha * X
      if (device.is_blas()) {
        vec::saxpy_blas(spatial_size, 1.0f, &bias_val, 0, out_ptr, 1);
      } else {
        for (int64_t k = 0; k < spatial_size; ++k) {
          out_ptr[k] += bias_val;
        }
      }
    }
  }
}

// 单个卷积核 (可能包含多个通道) 翻转 (180度翻转)
// in: 输入权重 [out_channel, in_channel, kernel_h, kernel_w]
// out: 翻转后的权重 [out_channel, in_channel, kernel_h, kernel_w]
template <typename T>
void flip_kernel(const T *in, const int64_t out_channel,
                 const int64_t in_channel, const int64_t kernel_h,
                 const int64_t kernel_w, T *out) {

  const auto kernel_size = kernel_h * kernel_w;

#pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < out_channel * in_channel; ++b) {
    const T *in_bt = in + b * kernel_size;
    T *out_bt = out + b * kernel_size;

    // 180度翻转：(i,j) -> (kernel_h-1-i, kernel_w-1-j)
    for (int64_t h = 0; h < kernel_h; ++h) {
      for (int64_t w = 0; w < kernel_w; ++w) {
        auto in_idx = h * kernel_w + w;
        auto out_idx = (kernel_h - 1 - h) * kernel_w + (kernel_w - 1 - w);
        out_bt[out_idx] = in_bt[in_idx];
      }
    }
  }
}

// 在已经经过填充输入数据（不包含批次）中插入零, 实现上采样,
// 用于卷积反向传播计算对 X 的梯度 in: 输入数据指针 shape: [channels, in_h,
// in_w] stride_h, stride_w：前向传播时卷积步长 pad_h, pad_w:
// 不是原始填充尺寸，是计算所得 out: 输入指针 out_h,out_w: 输出特征图高/宽
template <typename T>
void cnn_upsample_with_pad(const T *in, const int64_t channels,
                           const int64_t in_h, const int64_t in_w,
                           const int64_t stride_h, const int64_t stride_w,
                           const int64_t pad_h, const int64_t pad_w, T *out,
                           const int64_t out_h, const int64_t out_w) {

  const auto in_spatial_size = in_h * in_w;
  const auto out_spatial_size = out_h * out_w;

  if (stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0) {
    if (out_h == in_h && out_w == in_w) {
      // 输入与输出形状相同
      std::copy_n(in, channels * in_h * in_w, out);
      return;
    }
  }

  for (int64_t c = 0; c < channels; ++c) {
    auto in_bt = in + c * in_spatial_size;
    auto out_bt = out + c * out_spatial_size;

    // 只需要处理特征图
    for (int64_t h = 0; h < in_h; ++h) {
      for (int64_t w = 0; w < in_w; ++w) {
        // 计算新张量中的位置
        const auto new_h = h * stride_h + pad_h;
        const auto new_w = w * stride_w + pad_w;

        // 复制元素
        if (new_h >= 0 && new_h < out_h && new_w >= 0 && new_w < out_w) {
          out_bt[new_h * out_w + new_w] = in_bt[h * in_w + w];
        }
      }
    }
  }
}

// 为转置卷积展开权重矩阵
// 这个函数将权重从 [C_out, C_in, kernel_h, kernel_w] 重排为适合转置卷积的形式
// W: 输入权重，形状 [C_out, C_in, kernel_h, kernel_w]
// out: 输出权重矩阵，形状 [C_in * kernel_h * kernel_w, C_out]
template <typename T>
void transpose_conv_weight_reshape(const T *w, const int64_t out_channel,
                                   const int64_t in_channel,
                                   const int64_t kernel_h,
                                   const int64_t kernel_w, T *out) {

  const auto channel_kernel_size = in_channel * kernel_h * kernel_w;

  // 重排权重：从 [C_out, C_in, kernel_h, kernel_w] 到 [C_in * kernel_h *
  // kernel_w, C_out]
  for (int64_t c_out = 0; c_out < out_channel; ++c_out) {
    auto w_bt = w + c_out * channel_kernel_size;
    auto out_bt = out + c_out;
    for (int64_t n = 0; n < channel_kernel_size; ++n) {
      auto out_ptr = out_bt + n * out_channel;
      *out_ptr = w_bt[n];
    }
  }
}

// 包含批次的填充
// input: [N, C, H, W]
void pad(const dense::Tensor &input, const int64_t pad_h, const int64_t pad_w,
         dense::Tensor &output) {
  const auto N = input.size(0);
  const auto C = input.size(1);
  const auto H = input.size(2);
  const auto W = input.size(3);

  // 计算填充之后的特征维度
  const auto out_h = H + pad_h * 2;
  const auto out_w = W + pad_w * 2;

  if (out_h == H && out_w == W) {
    // 没有填充，我们直接用原始输入
    output = input.clone();
    return;
  }
  // 有填充，我们尝试用缓存
  std::vector<int64_t> pad_shape = {N, C, out_h, out_w};
  if (!output.is_defined() || output.sizes() != pad_shape) {
    output = dense::Tensor::zeros(input.dtype(), pad_shape);
  } else {
    output.zero_();
  }
  for (int64_t n = 0; n < N; ++n) {
    auto in_bt = input.const_data_as<float>() + n * C * H * W;
    auto out_bt = output.mutable_data_as<float>() + n * C * out_h * out_w;
    cnn_pad(in_bt, C, H, W, pad_h, pad_w, out_bt);
  }
}

// input: [N, C, H, W],填充之后的张量
// kernel_h,kernel_w：卷积核尺寸
void im2col(const dense::Tensor &input, const int64_t kernel_h,
            const int64_t kernel_w, const int64_t stride_h,
            const int64_t stride_w, dense::Tensor &output) {
  const auto N = input.size(0);
  const auto C = input.size(1);
  const auto H = input.size(2);
  const auto W = input.size(3);

  // 计算卷积输出特征维度的形状
  const int64_t out_h = (H - kernel_h) / stride_h + 1;
  const int64_t out_w = (W - kernel_w) / stride_w + 1;

  // 计算展平矩阵形状
  const int64_t col_h = out_h * out_w;
  const int64_t col_w = kernel_h * kernel_w * C;

  // 尝试使用缓存
  std::vector<int64_t> unfolded_shape = {N, col_h, col_w};
  if (!output.is_defined() || output.sizes() != unfolded_shape) {
    output = dense::Tensor::zeros(input.dtype(), unfolded_shape);
  } else {
    output.zero_();
  }

  for (int64_t n = 0; n < N; ++n) {
    auto in_bt = input.const_data_as<float>() + n * C * H * W;
    auto out_bt = output.mutable_data_as<float>() + n * col_h * col_w;
    cnn_im2col(in_bt, C, H, W, stride_h, stride_w, kernel_h, kernel_w, out_bt);
  }
}

} // namespace

// 用于卷积层前向传播计算辅助
class Conv2d::Conv2dHelper {
public:
  Conv2dHelper() = default;
  ~Conv2dHelper() = default;
  // input: [N, in_channel, H, W]
  // W: [out_channel, in_channel, kernel_h, kernel_w]
  // b: [out_channel]
  // stride_h,stride_w: 卷积步长
  // pad_h,pad_w: 填充尺寸
  dense::Tensor conv2d(const Device &device, const dense::Tensor &input,
                       const dense::Tensor &w, const dense::Tensor &b,
                       const int64_t stride_h, const int64_t stride_w,
                       const int64_t pad_h, const int64_t pad_w) {
    const auto out_channel = w.size(0);
    const auto in_channel = w.size(1);
    const auto kernel_h = w.size(2);
    const auto kernel_w = w.size(3);

    // 先 pad
    pad(input, pad_h, pad_w, pad_input_);
    // 展平
    im2col(pad_input_, kernel_h, kernel_w, stride_h, stride_w, unfolded_input_);

    const auto N = input.size(0);
    const auto padded_in_h = pad_input_.size(-2);
    const auto padded_in_w = pad_input_.size(-1);

    // 计算输出特征图形状
    const auto out_h =
        static_cast<int64_t>((padded_in_h - kernel_h) / stride_h) + 1;
    const auto out_w =
        static_cast<int64_t>((padded_in_w - kernel_w) / stride_w) + 1;

    auto output =
        dense::Tensor::zeros(input.dtype(), {N, out_channel, out_h, out_w});

    auto col_h = unfolded_input_.size(-2);
    auto col_w = unfolded_input_.size(-1);

    for (int64_t n = 0; n < N; ++n) {
      auto col_bt = unfolded_input_.const_data_as<float>() + n * col_h * col_w;
      auto out_bt =
          output.mutable_data_as<float>() + n * out_channel * out_h * out_w;
      cnn_conv2d(device, col_bt, col_h, col_w, w.const_data_as<float>(),
                 out_channel, in_channel, kernel_h, kernel_w,
                 b.const_data_as<float>(), out_bt, out_h, out_w);
    }

    return output;
  }

  // 用于反向传播计算对权重的梯度
  dense::Tensor unfolded_input() const { return unfolded_input_; }

private:
  dense::Tensor pad_input_;
  dense::Tensor unfolded_input_;
};

// 用于卷积层反向传播计算对输入的梯度
class Conv2d::ConvTranspose2dHelper {
public:
  ConvTranspose2dHelper() = default;
  ~ConvTranspose2dHelper() = default;

  // 使用 翻转卷积核，并应用 conv2d 实现的反向梯度计算
  // grad_output: 上游梯度,[N, out_channel, OH, OW]
  // 卷积核: [out_channel, in_channel, kernel_h, kernel_h]
  // stride_h,stride_w: 前向传播时的卷积步长
  // pad_h,pad_w: 前向传播时的填充尺寸
  // in_h,in_w: 前向传播输入的特征图尺寸
  dense::Tensor conv_transpose2d(const Device &device,
                                 const dense::Tensor &grad_output,
                                 const dense::Tensor &w, const int64_t stride_h,
                                 const int64_t stride_w, const int64_t pad_h,
                                 const int64_t pad_w, const int64_t in_h,
                                 const int64_t in_w) {
    // 输入是梯度：grad_output [N, C_out, H_out, W_out]
    const auto N = grad_output.size(0);
    const auto out_channel = grad_output.size(1);
    const auto out_h = grad_output.size(2);
    const auto out_w = grad_output.size(3);

    const auto in_channel = w.size(1);
    const auto kernel_h = w.size(2);
    const auto kernel_w = w.size(3);

    // 计算有效填充
    const auto effective_pad_h = kernel_h - 1 - pad_h;
    const auto effective_pad_w = kernel_w - 1 - pad_w;

    // 计算上采样的尺寸
    const auto upsampled_h = out_h + (out_h - 1) * (stride_h - 1);
    const auto upsampled_w = out_w + (out_w - 1) * (stride_w - 1);

    // 最终经过填充和上采样的形状
    const auto final_h = upsampled_h + 2 * effective_pad_h;
    const auto final_w = upsampled_w + 2 * effective_pad_w;

    std::vector<int64_t> upsampled_final_shape = {N, out_channel, final_h,
                                                  final_w};
    if (!upsampled_padded_grad_output_.is_defined() ||
        upsampled_padded_grad_output_.sizes() != upsampled_final_shape) {
      upsampled_padded_grad_output_ =
          dense::Tensor::zeros(grad_output.dtype(), upsampled_final_shape);
    } else {
      upsampled_padded_grad_output_.zero_();
    }

    for (int64_t n = 0; n < N; ++n) {
      auto grad_bt =
          grad_output.const_data_as<float>() + n * out_channel * out_h * out_w;
      auto out_bt = upsampled_padded_grad_output_.mutable_data_as<float>() +
                    n * out_channel * final_h * final_w;
      cnn_upsample_with_pad(grad_bt, out_channel, out_h, out_w, stride_h,
                            stride_w, effective_pad_h, effective_pad_w, out_bt,
                            final_h, final_w);
    }

    // 展开
    im2col(upsampled_padded_grad_output_, kernel_h, kernel_w, 1, 1,
           unflolded_grad_output_);

    auto flip_w = dense::Tensor::zeros_like(w);
    // 将卷积核翻转180度
    flip_kernel(w.const_data_as<float>(), out_channel, in_channel, kernel_h,
                kernel_w, flip_w.mutable_data_as<float>());
    auto transpose_w = flip_w.transpose(0, 1);

    auto output =
        dense::Tensor::zeros(grad_output.dtype(), {N, in_channel, in_h, in_w});

    auto col_h = unflolded_grad_output_.size(-2);
    auto col_w = unflolded_grad_output_.size(-1);

    for (int64_t n = 0; n < N; ++n) {
      auto col_bt =
          unflolded_grad_output_.const_data_as<float>() + n * col_h * col_w;
      auto out_bt =
          output.mutable_data_as<float>() + n * in_channel * in_h * in_w;
      cnn_conv2d(device, col_bt, col_h, col_w,
                 transpose_w.const_data_as<float>(), transpose_w.size(0),
                 transpose_w.size(1), transpose_w.size(2), transpose_w.size(3),
                 nullptr, out_bt, in_h, in_w);
    }

    return output;
  }

  // 另外一种计算反向输入梯度的等效方法
  // 使用 im2col, matmul, col2im 方法
  dense::Tensor conv_transpose2d_col2im(
      const Device &deivce, const dense::Tensor &grad_output,
      const dense::Tensor &w, const int64_t stride_h, const int64_t stride_w,
      const int64_t pad_h, const int64_t pad_w, const int64_t in_h,
      const int64_t in_w) {
    // 输入梯度：grad_output [N, C_out, H_out, W_out]
    const auto N = grad_output.size(0);
    const auto out_channel = grad_output.size(1);
    const auto out_h = grad_output.size(2);
    const auto out_w = grad_output.size(3);

    // 权重：W [C_out, C_in, kernel_h, kernel_w]
    const auto in_channel = w.size(1);
    const auto kernel_h = w.size(2);
    const auto kernel_w = w.size(3);
    const auto kernel_size = kernel_h * kernel_w;

    // 步骤1：将grad_output进行im2col展开
    // 对于转置卷积，我们需要将grad_output当作"卷积核"来展开
    const auto spatial_size = out_h * out_w;

    // 使用现有的im2col，但这里我们把grad_output当作输入来展开
    // 展开后的形状：[B, spatial_size, C_out]
    im2col(grad_output, 1, 1, 1, 1,
           unflolded_grad_output_); // 使用1x1卷积核展开

    // 步骤2：重排权重矩阵
    // 将权重从 [C_out, C_in, K_H, K_W] 重排为 [C_in * K_H * K_W, C_out]
    std::vector<int64_t> weight_matrix_shape = {in_channel * kernel_size,
                                                out_channel};
    if (!col_matrix_.is_defined() ||
        col_matrix_.sizes() != weight_matrix_shape) {
      col_matrix_ = dense::Tensor::zeros(w.dtype(), weight_matrix_shape);
    }

    transpose_conv_weight_reshape(w.const_data_as<float>(), out_channel,
                                  in_channel, kernel_h, kernel_w,
                                  col_matrix_.mutable_data_as<float>());

    // 步骤3：矩阵乘法
    // grad_output_unfolded [B, spatial_size, C_out] × weight_matrix [C_out,
    // C_in
    // * K_H * K_W] 结果: [B, spatial_size, C_in * K_H * K_W]

    std::vector<int64_t> matmul_result_shape = {N, spatial_size,
                                                in_channel * kernel_size};
    auto matmul_result =
        Tensor::zeros(grad_output.dtype(), matmul_result_shape);

    auto grad_ptr = unflolded_grad_output_.const_data_as<float>();
    auto weight_ptr = col_matrix_.const_data_as<float>();
    auto result_ptr = matmul_result.mutable_data_as<float>();

    // 对每个batch进行矩阵乘法
    for (int64_t n = 0; n < N; ++n) {
      const float *grad_bt = grad_ptr + n * spatial_size * out_channel;
      float *result_bt =
          result_ptr + n * spatial_size * in_channel * kernel_size;

      // grad_bt: [spatial_size, C_out]
      // weight_ptr: [C_in * kernel_size, C_out] (需要转置使用)
      // result_bt: [spatial_size, C_in * kernel_size]
      if (deivce.is_blas()) {
        vec::matmul_B_transpose_blas(
            grad_bt, out_channel,    // A: [spatial_size, C_out]
            weight_ptr, out_channel, // B: [C_in * kernel_size, C_out]
            nullptr,                 // no bias
            result_bt,
            in_channel * kernel_size, // C: [spatial_size, C_in * kernel_size]
            spatial_size, in_channel * kernel_size, out_channel);
      } else {
        vec::matmul_B_transpose_native(
            grad_bt, out_channel,    // A: [spatial_size, C_out]
            weight_ptr, out_channel, // B: [C_in * kernel_size, C_out]
            nullptr,                 // no bias
            result_bt,
            in_channel * kernel_size, // C: [spatial_size, C_in * kernel_size]
            spatial_size, in_channel * kernel_size, out_channel);
      }
    }

    // 步骤4：col2im - 将矩阵结果重新折叠成特征图
    auto output =
        Tensor::zeros(grad_output.dtype(), {N, in_channel, in_h, in_w});

    for (int64_t n = 0; n < N; ++n) {
      auto col_bt = matmul_result.const_data_as<float>() +
                    n * spatial_size * in_channel * kernel_size;
      auto out_bt =
          output.mutable_data_as<float>() + n * in_channel * in_h * in_w;
      cnn_col2im(col_bt, in_channel, kernel_h, kernel_w, stride_h, stride_w,
                 pad_h, pad_w, out_bt, in_h, in_w);
    }

    return output;
  }

private:
  Tensor upsampled_padded_grad_output_;
  Tensor unflolded_grad_output_;
  Tensor col_matrix_;
};

Conv2d::Conv2d(Context *ctx, const std::string &name, int64_t in_channels,
               int64_t out_channels, int64_t kernel_h, int64_t kernel_w,
               int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w,
               bool has_bias)
    : Layer(ctx, name), in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h),
      stride_w_(stride_w), pad_h_(pad_h), pad_w_(pad_w), has_bias_(has_bias),
      conv2d_(new Conv2dHelper()),
      conv_transpose_2d_(new ConvTranspose2dHelper()) {
  RegisterParam();
}

Conv2d::~Conv2d() = default;

void Conv2d::init() {
  W_ = dense::Tensor::empty(
      DType::kFloat32, {out_channels_, in_channels_, kernel_h_, kernel_w_});
  init::kaiming_normal_(W_, std::sqrt(5));
  if (has_bias_) {
    b_ = dense::Tensor::zeros(DType::kFloat32, {out_channels_});
    auto fan = init::_calculate_fan_in_and_fan_out(W_);
    auto fan_in = std::get<0>(fan);
    auto bound = fan_in > 0 ? 1.0 / std::sqrt(fan_in) : 0;
    init::uniform_(b_, -bound, bound);
  }
}

dense::Tensor Conv2d::forward(const dense::Tensor &input) {
  // 输入形状： (N, C, H, W)
  // 输出形状： (N, OC, OH, OW)

  in_shape_ = input.sizes();

  auto output = conv2d_->conv2d(ctx()->device, input, W_, b_, stride_h_,
                                stride_w_, pad_h_, pad_w_);

  return output;
}

dense::Tensor Conv2d::backward(const dense::Tensor &grad_output) {
  const auto N = grad_output.size(0);
  const auto OC = grad_output.size(1);
  const auto OH = grad_output.size(2);
  const auto OW = grad_output.size(3);

  // grad_output (N, out_channel, OH, OW)
  if (!grad_W_.is_defined()) {
    // grad_W_ 的形状: [out_channel, in_channel, kernel_h, kernel_w]
    // (权重梯度矩阵)
    grad_W_ = dense::Tensor::zeros_like(W_);
  }
  if (has_bias_ && !grad_b_.is_defined()) {
    // grad_b_ 的形状: [out_channel] (偏置梯度向量)
    grad_b_ = dense::Tensor::zeros_like(b_);
  }

  // 1. 计算偏置梯度 (grad_b)
  // 输出特征图尺寸
  const auto spatial_size = OH * OW;

  auto grad_out_ptr = grad_output.const_data_as<float>();

  if (has_bias_) {
    auto grad_b_ptr = grad_b_.mutable_data_as<float>();

    for (int64_t n = 0; n < N; ++n) {
      for (int64_t c = 0; c < OC; ++c) {
        auto grad_bt = grad_out_ptr + n * OC * spatial_size + c * spatial_size;
        for (int64_t i = 0; i < spatial_size; ++i) {
          grad_b_ptr[c] += grad_bt[i];
        }
      }
    }
  }
  // 或者更高效的实现
  /*
  for (int64_t idx = 0; idx < grad_output.numel(); ++idx) {
    const auto c = (idx / spatial_size) % OC;
    grad_b_ptr[c] += grad_out_ptr[idx];
  }*/

  // 2. 计算权重梯度 (grad_W)
  /*
  我们在前向传播时，通过矩阵乘法一次性计算出输出，其计算公式为：
  Z = W*Y^T
  这里的 Y 就是 unfoled_input
  那么根据这个公式，计算反向传播 dL/dW 的梯度：
  dL/dW = dL/dZ * Y
  具体还需要在批次维度上求和，因为每个批次 W 都参与了计算。
  设 T = dL/dZ * Y
  我们来分析形状：
  dL/dZ 将其重塑为 [N, out_channel,OH*OW]
  Y 的形状为 [N, OH*OW, in_channel*kernel_h*kernel_w]
  则 T 的形状为 [N, out_channel, in_channel*kernel_h*kernel_w]
  在 T 的批次维度求和，最终得到 dL/dW,
  其形状为[out_channel,in_channel*kernel_h*kernel_w] 最终重塑为 [out_channel,
  in_channel,kernel_h, kernel_w]
  */
  const auto in_channel = W_.size(1);
  const auto kernel_h = W_.size(2);
  const auto kernel_w = W_.size(3);

  // 中间步骤，先计算 T
  // kernel_spatial_size = unfolded_input 的最后一个维度
  const auto kernel_spatial_size = in_channel * kernel_h * kernel_w;

  // matmul_result 为下面矩阵乘法的输出，也就是注释中提到的 'T'
  auto matmul_result =
      dense::Tensor::zeros(grad_output.dtype(), {N, OC, kernel_spatial_size});
  auto unfolded_input = conv2d_->unfolded_input();

  // unfolded_input: [N, OH*OW, kernel_spatial_size]
  // 这里与其他深度学习框架可能不同，也就是一个转置的关系，就看 im2col 如何实现
  const auto unfoled_h = unfolded_input.size(1);
  const auto unfoled_w = unfolded_input.size(2);

  assert(unfoled_h == OH * OW);

  for (int64_t n = 0; n < N; ++n) {
    auto grad_out_bt = grad_out_ptr + n * OC * OH * OW;
    auto unfolded_in_bt =
        unfolded_input.const_data_as<float>() + n * unfoled_h * unfoled_w;
    auto matmul_result_bt =
        matmul_result.mutable_data_as<float>() + n * OC * kernel_spatial_size;
    if (ctx()->device.is_blas()) {
      vec::matmul_blas(grad_out_bt, OH * OW, unfolded_in_bt, unfoled_w, nullptr,
                       matmul_result_bt, kernel_spatial_size, OC, OH * OW,
                       unfoled_w);
    } else {
      vec::matmul_native(grad_out_bt, OH * OW, unfolded_in_bt, unfoled_w,
                         nullptr, matmul_result_bt, kernel_spatial_size, OC,
                         OH * OW, unfoled_w);
    }
  }
  auto grad_w_ptr = grad_W_.mutable_data_as<float>();
  // 在批次维度求和
  for (int64_t n = 0; n < N; ++n) {
    auto matmul_result_bt =
        matmul_result.const_data_as<float>() + n * OC * kernel_spatial_size;
    for (size_t i = 0; i < grad_W_.numel(); ++i) {
      grad_w_ptr[i] += matmul_result_bt[i];
    }
  }

  // 3. 计算对输入的梯度，这个梯度回传到上一层
  auto output = conv_transpose_2d_->conv_transpose2d(
      ctx()->device, grad_output, W_, stride_h_, stride_w_, pad_h_, pad_w_,
      in_shape_[2], in_shape_[3]);
  return output;
}

} // namespace dense