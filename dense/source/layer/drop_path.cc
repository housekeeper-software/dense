#include "layer/drop_path.h"
#include "layer/init.h"
#include "math/vec_math.h"
#include <assert.h>

namespace dense {

DropPath::DropPath(Context *ctx, const std::string &name, float drop_prob)
    : Layer(ctx, name), drop_prob_(drop_prob) {
  assert(drop_prob >= 0.0f && drop_prob <= 1.0f);
}

Tensor DropPath::forward(const Tensor &input) {
  // 推理模式或drop_prob为0时，直接返回输入
  if (!is_training() || drop_prob_ == 0.0f) {
    return input.clone();
  }

  // 如果drop_prob为1，返回全零张量
  if (drop_prob_ == 1.0f) {
    return dense::Tensor::zeros_like(input);
  }

  const auto B = input.size(0);

  drop_mask_ = dense::Tensor::empty(dense::DType::kFloat32, {B});
  float keep_prob = 1.0f - drop_prob_;
  float scale = 1.0f / keep_prob; // 缩放因子，补偿被丢弃的路径

  // 生成伯努利掩码
  init::bernoulli_(drop_mask_, keep_prob);

  // 对掩码进行缩放
  auto mask_ptr = drop_mask_.mutable_data_as<float>();
  for (size_t i = 0; i < drop_mask_.numel(); ++i) {
    mask_ptr[i] *= scale;
  }

  auto output = Tensor::empty(input.dtype(), input.sizes());
  auto input_ptr = input.const_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();

  const auto elements_per_batch = input.numel() / B;

  for (int64_t b = 0; b < B; ++b) {
    float mask_val = mask_ptr[b];
    auto in_bt = input_ptr + b * elements_per_batch;
    auto out_bt = out_ptr + b * elements_per_batch;
    if (ctx()->device.is_blas()) {
      vec::scopy_blas(elements_per_batch, in_bt, 1, out_bt, 1);
      vec::sscal_blas(elements_per_batch, mask_val, out_bt, 1);
    } else {
      // 原生实现
      for (size_t i = 0; i < elements_per_batch; ++i) {
        out_bt[i] = in_bt[i] * mask_val;
      }
    }
  }
  return output;
}

Tensor DropPath::backward(const Tensor &grad_output) {
  if (!is_training() || drop_prob_ == 0.0f) {
    return grad_output.clone();
  }
  // 如果drop_prob为1，返回全零梯度
  if (drop_prob_ == 1.0f) {
    return dense::Tensor::zeros_like(grad_output);
  }

  const auto B = grad_output.size(0);
  auto grad_input = Tensor::empty(grad_output.dtype(), grad_output.sizes());

  auto grad_output_ptr = grad_output.const_data_as<float>();
  auto grad_input_ptr = grad_input.mutable_data_as<float>();
  auto mask_ptr = drop_mask_.const_data_as<float>();

  const auto elements_per_batch = grad_output.numel() / B;

  for (size_t b = 0; b < B; ++b) {
    float mask_val = mask_ptr[b];
    auto g_out_bt = grad_output_ptr + b * elements_per_batch;
    auto g_in_bt = grad_input_ptr + b * elements_per_batch;

    if (ctx()->device.is_blas()) {
      // 使用BLAS向量缩放
      vec::scopy_blas(elements_per_batch, g_out_bt, 1, g_in_bt, 1);
      vec::sscal_blas(elements_per_batch, mask_val, g_in_bt, 1);
    } else {
      // 原生实现
      for (size_t i = 0; i < elements_per_batch; ++i) {
        g_in_bt[i] = g_out_bt[i] * mask_val;
      }
    }
  }

  return grad_input;
}
} // namespace dense