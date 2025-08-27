#include "layer/patch_embed.h"
#include "layer/conv2d.h"
#include "layer/dropout.h"
#include "layer/embedding.h"
#include "layer/flatten.h"
#include "layer/init.h"
#include "math/vec_math.h"
#include <cmath>

namespace dense {

PatchEmbed::PatchEmbed(Context *ctx, const std::string &name,
                       int64_t hidden_size, int64_t image_size,
                       int64_t patch_size, int64_t num_channels, bool bias)
    : Layer(ctx, name), hidden_size_(hidden_size), image_size_(image_size),
      patch_size_(patch_size), num_channels_(num_channels), num_patches_(0) {
  RegisterParam();
  num_patches_ = (image_size_ / patch_size_) * (image_size_ / patch_size_);
  conv2d_ = std::make_unique<Conv2d>(
      ctx, make_layer_name("%s.conv2d", name.c_str()), num_channels_,
      hidden_size_, patch_size, patch_size, patch_size, patch_size, 0, 0, bias);
  flatten_ = std::make_unique<Flatten>(
      ctx, make_layer_name("%s.flatten", name.c_str()), 2, -1);
  pos_embedding_ =
      std::make_unique<Embedding>(ctx, make_layer_name("%s.wpe", name.c_str()),
                                  num_patches_ + 1, hidden_size_);
}

PatchEmbed::~PatchEmbed() = default;

void PatchEmbed::init() {
  W_ = Tensor::empty(DType::kFloat32, {hidden_size_});
  init::normal_(W_);
}

dense::Tensor PatchEmbed::forward(const dense::Tensor &input) {
  // input [B, 1, 28,28]
  auto x = conv2d_->forward(input);
  // x [ B,768, 7,7]
  x = flatten_->forward(x);
  // x [B,768,49]
  x = x.transpose(1, 2);
  // x [B,49,768]

  const auto B = x.size(0);
  const auto T = x.size(1);
  const auto C = x.size(2);

  const auto T1 = T + 1;

  auto output = dense::Tensor::empty(x.dtype(), {B, T1, C});

  auto x_ptr = x.const_data_as<float>();
  auto out_ptr = output.mutable_data_as<float>();

  for (int64_t b = 0; b < B; ++b) {
    auto x_bt = x.const_data_as<float>() + b * T * C;
    auto out_bt = out_ptr + b * T1 * C;
    vec::scopy_blas(C, W_.const_data_as<float>(), 1, out_bt, 1);
    out_bt += C;
    vec::scopy_blas(T * C, x_bt, 1, out_bt, 1);
  }

  std::vector<int64_t> pos_data(B * T1);
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T1; ++t) {
      pos_data[b * T1 + t] = static_cast<int64_t>(t);
    }
  }

  auto pos =
      dense::Tensor::from_blob(dense::DType::kInt64, {B, T1}, &pos_data[0]);
  auto pos_emb = pos_embedding_->forward(pos);

  {
    auto N = output.numel();
    auto tok_ptr = output.mutable_data_as<float>();
    auto pos_ptr = pos_emb.const_data_as<float>();
    if (ctx()->device.is_blas()) {
      vec::saxpy_blas(N, 1.0f, pos_ptr, 1, tok_ptr, 1);
    } else {
      for (size_t i = 0; i < N; ++i) {
        tok_ptr[i] = tok_ptr[i] + pos_ptr[i];
      }
    }
  }
  return output;
}

dense::Tensor PatchEmbed::backward(const dense::Tensor &grad_output) {
  pos_embedding_->backward(grad_output);

  const auto B = grad_output.size(0);
  const auto T = grad_output.size(1);
  const auto C = grad_output.size(2);

  if (!grad_W_.is_defined()) {
    grad_W_ = Tensor::zeros_like(W_);
  }
  auto grad_w_ptr = grad_W_.mutable_data_as<float>();
  auto grad_out_ptr = grad_output.const_data_as<float>();

  const auto T1 = T - 1;

  auto grad_input = Tensor::empty(grad_output.dtype(), {B, T1, C});
  auto grad_in_ptr = grad_input.mutable_data_as<float>();

  for (int64_t b = 0; b < B; ++b) {
    auto grad_out_bt = grad_out_ptr + b * T * C;
    auto grad_in_bt = grad_in_ptr + b * T1 * C;

    if (ctx()->device.is_blas()) {
      vec::saxpy_blas(C, 1.0f, grad_out_bt, 1, grad_w_ptr, 1);
    } else {
      for (int64_t k = 0; k < C; ++k) {
        grad_w_ptr[k] += grad_out_bt[k];
      }
    }
    grad_out_bt += C;
    vec::scopy_blas(T1 * C, grad_out_bt, 1, grad_in_bt, 1);
  }
  grad_input = grad_input.transpose(1, 2);
  grad_input = flatten_->backward(grad_input);
  return conv2d_->backward(grad_input);
}

} // namespace dense