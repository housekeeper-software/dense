#include "layer/multi_head_attention.h"
#include "layer/dropout.h"
#include "layer/init.h"
#include "layer/linear.h"
#include "math/vec_math.h"
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

namespace dense {

namespace {

void split_qkv(const dense::Tensor &A, dense::Tensor &q, dense::Tensor &k,
               dense::Tensor &v) {
  const auto B = A.size(0);
  const auto T = A.size(1);
  const auto C = A.size(2);

  const int64_t dim = C / 3;
  std::vector<int64_t> shape = {B, T, dim};

  if (!q.is_defined() || q.sizes() != shape) {
    q = dense::Tensor::empty(A.dtype(), shape);
  }
  if (!k.is_defined() || k.sizes() != shape) {
    k = dense::Tensor::empty(A.dtype(), shape);
  }
  if (!v.is_defined() || v.sizes() != shape) {
    v = dense::Tensor::empty(A.dtype(), shape);
  }

  // 每个特征向量包含的数据长度，字节单位
  const int64_t data_size = A.element_size() * dim;

  auto A_ptr = A.const_data_as<float>();
  auto q_ptr = q.mutable_data_as<float>();
  auto k_ptr = k.mutable_data_as<float>();
  auto v_ptr = v.mutable_data_as<float>();

  for (int64_t i = 0; i < B * T; ++i) {
    auto a_bt = A_ptr + i * C;

    std::memcpy(q_ptr + i * dim, a_bt, data_size);
    std::memcpy(k_ptr + i * dim, a_bt + dim, data_size);
    std::memcpy(v_ptr + i * dim, a_bt + 2 * dim, data_size);
  }
}
} // namespace

MultiHeadAttention::MultiHeadAttention(Context *ctx, const std::string &name,
                                       int64_t n_heads, int64_t emb_dim,
                                       int64_t context_length, bool bias,
                                       float drop_rate, bool use_attn_mask,
                                       std::shared_ptr<LayerCache> cache)
    : Layer(ctx, name), head_dim_(0), n_heads_(n_heads), emb_dim_(emb_dim),
      context_length_(context_length), bias_(bias), drop_rate_(drop_rate),
      cache_(cache), attn_scale_(0.0f) {
  assert(emb_dim_ % n_heads_ == 0);

  // 每个头的维度
  head_dim_ = emb_dim_ / n_heads_;

  // 缩放因子
  attn_scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

  in_proj_ =
      std::make_unique<Linear>(ctx, make_layer_name("%s.c_attn", name.c_str()),
                               emb_dim_, 3 * emb_dim_, bias_);

  out_proj_ =
      std::make_unique<Linear>(ctx, make_layer_name("%s.c_proj", name.c_str()),
                               emb_dim_, emb_dim_, bias_);

  // nn.MultiheadAttention 是这么初始化的
  init::xavier_uniform_(in_proj_->W_);

  if (in_proj_->b_.is_defined()) {
    in_proj_->b_.zero_();
  }

  if (out_proj_->b_.is_defined()) {
    out_proj_->b_.zero_();
  }

  if (use_attn_mask) {
    attn_mask_ =
        Tensor::zeros(DType::kInt8, {context_length_, context_length_});
    // 生成一个上三角掩码矩阵
    // 1 的地方需要掩码，就是将注意力矩阵的对应位置元素设置为 -inf
    // 这样经过 softmax 之后，这些位置都变成了 0
    auto ptr = attn_mask_.mutable_data_as<int8_t>();
    auto M = attn_mask_.size(0);
    auto N = attn_mask_.size(1);
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = m + 1; n < N; ++n) {
        ptr[m * N + n] = 1;
      }
    }
  }
}

MultiHeadAttention::~MultiHeadAttention() = default;

dense::Tensor MultiHeadAttention::forward(const dense::Tensor &input) {
  const auto B = input.size(0);
  const auto T = input.size(1);
  const auto C = input.size(2);

  auto qkv = in_proj_->forward(input);
  split_qkv(qkv, q_, k_, v_);
  auto q = q_;
  auto k = k_;
  auto v = v_;

  if (!ctx()->training && cache_) {
    // 在推理场景，并且启用了 kv_cache，
    // 先将 k, v 更新到 cache 中
    // 再将 k, v 从 cache 中取出
    cache_->update(k, v);
    k = cache_->key_states();
    v = cache_->value_states();
    // 此刻，q,k,v 的形状不再相同
  }

  // 此刻，k, v 的形状一致，但是与 q 的形状不完全一致
  // 在训练过程，或者未启用 kv_cache，T == total_seq_len
  const auto total_seq_len = k.size(1);

  // 每个头att的形状都是 [T,total_seq_len],用于保存中间计算结果
  auto att = dense::Tensor::zeros(dense::DType::kFloat32, {T, total_seq_len});

  if (is_training()) {
    // 缓存 softmax 的输出，因为 softmax 反向传播时依赖前向输出
    att_softmax_output_ = dense::Tensor::zeros(dense::DType::kFloat32,
                                               {B, n_heads_, T, total_seq_len});
    // 缓存 dropout 的输出
    att_dropout_output_ = dense::Tensor::zeros(dense::DType::kFloat32,
                                               {B, n_heads_, T, total_seq_len});

    attn_dropout_mask_ = dense::Tensor::empty(dense::DType::kFloat32,
                                              {B, n_heads_, T, total_seq_len});

    double plm = 1.0 - drop_rate_;
    // 防止分母为 0
    float scale = plm == 0 ? 0.0f : 1.0 / plm;
    // bernoulli 分布，plm 是保留概率
    init::bernoulli_(attn_dropout_mask_, plm);
    auto attn_dropout_mask_ptr = attn_dropout_mask_.mutable_data_as<float>();
    for (size_t i = 0; i < attn_dropout_mask_.numel(); ++i) {
      attn_dropout_mask_ptr[i] *= scale;
    }
  }

  // 输出张量
  auto output = dense::Tensor::zeros_like(input);

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < n_heads_; ++h) {
      if (ctx()->device.is_blas()) {
        header_forward_blas(q, k, v, output, att, b, h);
      } else {
        header_forward_native(q, k, v, output, att, b, h);
      }
    }
  }
  return out_proj_->forward(output);
}

dense::Tensor MultiHeadAttention::backward(const dense::Tensor &grad_output) {
  // grad_output 形状: [B,T,C]
  auto grad_input = out_proj_->backward(grad_output);

  auto B = grad_input.size(0);
  auto T = grad_input.size(1);
  auto C = grad_input.size(2);

  std::vector<int64_t> qkv_shape = {B, T, C * 3};

  auto grad_qkv = dense::Tensor::zeros(grad_input.dtype(), qkv_shape);
  // 用于保存中间计算结果，最终会合并到 grad_qkv
  auto grad_att = dense::Tensor::zeros(dense::DType::kFloat32, {T, T});

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < n_heads_; ++h) {
      if (ctx()->device.is_blas()) {
        header_backward_blas(q_, k_, v_, grad_qkv, grad_input, grad_att, b, h);
      } else {
        header_backward_native(q_, k_, v_, grad_qkv, grad_input, grad_att, b,
                               h);
      }
    }
  }
  return in_proj_->backward(grad_qkv);
}

void MultiHeadAttention::header_forward_native(const Tensor &q, const Tensor &k,
                                               const Tensor &v, Tensor &out,
                                               dense::Tensor &att, size_t b,
                                               size_t h) {
  // att 的形状 [T, total_seq_len]
  // 在推理过程中，如果启用 kv_cache，那么在第二次自回归推理: T=1,但
  // total_seq_len 是从第一次推理开始的总的 token 长度
  // 此刻，q 的形状可能是 [B,T,embedding_dim]
  // 但 k,v 的形状是 [B, total_seq_len , embedding_dim]
  // 在训练过程中，或者不使用 kv_cache，T == total_seq_len

  const auto T = att.size(0);
  const auto total_seq_len = att.size(1);

  auto q_bt = q.const_data_as<float>() + b * T * q.size(-1);
  auto k_bt = k.const_data_as<float>() + b * total_seq_len * k.size(-1);
  auto v_bt = v.const_data_as<float>() + b * total_seq_len * v.size(-1);

  att.zero_(); // att 清零，矩阵乘法输出是累加，不是赋值
  auto att_ptr = att.mutable_data_as<float>();

  // 当前批次的输出起始位置
  auto out_bt = out.mutable_data_as<float>() + b * T * out.size(-1);
  auto mask_ptr = attn_mask_.const_data_as<int8_t>();

  // 1. Q*K^T, [T,head_dim_] @ [total_seq_len,head_dim_]^T --> [T,total_seq_len]
  vec::matmul_B_transpose_native(q_bt + h * head_dim_, // Q 的起始位置
                                 q.size(-1),           // Q 的行距
                                 k_bt + h * head_dim_, // K 的起始位置
                                 k.size(-1),           // K 的行距
                                 nullptr,              // bias
                                 att_ptr,              // att:输出结果
                                 total_seq_len,        // att 的行距
                                 T,                    // M
                                 total_seq_len,        // N
                                 head_dim_             // K
  );

  // 对 att 逐元素缩放
  for (size_t i = 0; i < att.numel(); ++i) {
    att_ptr[i] *= attn_scale_; // 缩放
  }

  if (attn_mask_.is_defined()) {
    // 对 att[T,total_seq_len] 应用掩码
    // 掩码矩阵是 [config_.context_length,config_.context_length]
    auto mask_ptr = attn_mask_.const_data_as<int8_t>();
    const auto col = attn_mask_.size(-1);
    for (size_t i = total_seq_len - T; i < total_seq_len; ++i) {
      for (size_t j = 0; j < total_seq_len; ++j) {
        if (mask_ptr[i * col + j] == 1) {
          att_ptr[i * total_seq_len + j] = -INFINITY;
        }
      }
    }
  }
  // 计算 att 的 softmax
  vec::mat_softmax_forward_native(att_ptr, T, total_seq_len);

  if (is_training()) {
    // 缓存 softmax结算结果，在 softmax backward时需要用到
    auto ptr = att_softmax_output_.mutable_data_as<float>() +
               b * n_heads_ * T * total_seq_len +
               h * T * total_seq_len;        // 每个批次每个头的起始位置
    std::memcpy(ptr, att_ptr, att.nbytes()); // 缓存softmax的注意力
  }

  // 在推理场景下，dropout 不生效
  // drop_output[T,total_seq_len]
  if (attn_dropout_mask_.is_defined()) {

    auto mask_ptr = attn_dropout_mask_.const_data_as<float>() +
                    b * n_heads_ * T * total_seq_len + h * T * total_seq_len;

    for (size_t i = 0; i < att.numel(); ++i) {
      att_ptr[i] = att_ptr[i] * mask_ptr[i];
    }
    auto ptr = att_dropout_output_.mutable_data_as<float>() +
               b * n_heads_ * T * total_seq_len +
               h * T * total_seq_len; // 每个批次每个头的起始位置
                                      // 缓存 dropout 后的 attention
    std::memcpy(ptr, att_ptr, att.nbytes());
  }

  // 计算 attention @ V
  // 然后再直接赋值到输出的指定位置，这里也包含将多头合并到最终的输出
  vec::matmul_native(att_ptr, total_seq_len, v_bt + h * head_dim_, v.size(-1),
                     nullptr, out_bt + h * head_dim_, out.size(-1), T /*M*/,
                     total_seq_len /*K*/, head_dim_ /*N*/);
}

void MultiHeadAttention::header_forward_blas(const Tensor &q, const Tensor &k,
                                             const Tensor &v, Tensor &out,
                                             dense::Tensor &att, size_t b,
                                             size_t h) {
  const auto T = att.size(0);
  const auto total_seq_len = att.size(1);

  auto q_bt = q.const_data_as<float>() + b * T * q.size(-1);
  auto k_bt = k.const_data_as<float>() + b * total_seq_len * k.size(-1);
  auto v_bt = v.const_data_as<float>() + b * total_seq_len * v.size(-1);

  att.zero_(); // att 清零，矩阵乘法输出是累加，不是赋值
  auto att_ptr = att.mutable_data_as<float>();

  // 当前批次的输出起始位置
  auto out_bt = out.mutable_data_as<float>() + b * T * out.size(-1);

  // 1. Q*K^T, [T,head_dim_] @ [total_seq_len,head_dim_]^T --> [T,total_seq_len]
  vec::matmul_B_transpose_blas(q_bt + h * head_dim_, // Q 的起始位置
                               q.size(-1),           // Q 的行距
                               k_bt + h * head_dim_, // K 的起始位置
                               k.size(-1),           // K 的行距
                               nullptr,              // bias
                               att_ptr,              // att:输出结果
                               total_seq_len,        // att 的行距
                               T,                    // M
                               total_seq_len,        // N
                               head_dim_             // K
  );

  // 对 att 逐元素缩放
  vec::sscal_blas(att.numel(), attn_scale_, att_ptr, 1);

  if (attn_mask_.is_defined()) {
    // 对 att[T,total_seq_len] 应用掩码
    // 掩码矩阵是 [config_.context_length,config_.context_length]
    auto mask_ptr = attn_mask_.const_data_as<int8_t>();
    const auto col = attn_mask_.size(-1);
    for (size_t i = total_seq_len - T; i < total_seq_len; ++i) {
      for (size_t j = 0; j < total_seq_len; ++j) {
        if (mask_ptr[i * col + j] == 1) {
          att_ptr[i * total_seq_len + j] = -INFINITY;
        }
      }
    }
  }
  // 计算 att 的 softmax

  vec::mat_softmax_forward_blas(att_ptr, T, total_seq_len);

  if (is_training()) {
    // 缓存 softmax结算结果，在 softmax backward时需要用到
    auto ptr = att_softmax_output_.mutable_data_as<float>() +
               b * n_heads_ * T * total_seq_len +
               h * T * total_seq_len;        // 每个批次每个头的起始位置
    std::memcpy(ptr, att_ptr, att.nbytes()); // 缓存softmax的注意力
  }

  // 在推理场景下，dropout 不生效
  // drop_output[T,total_seq_len]
  if (attn_dropout_mask_.is_defined()) {
    auto mask_ptr = attn_dropout_mask_.const_data_as<float>() +
                    b * n_heads_ * T * total_seq_len + h * T * total_seq_len;
    vec::shdm_blas(att.numel(), att_ptr, mask_ptr, att_ptr);
    // 每个批次每个头的起始位置
    auto ptr = att_dropout_output_.mutable_data_as<float>() +
               b * n_heads_ * T * total_seq_len + h * T * total_seq_len;
    // 缓存 dropout 后的 attention
    std::memcpy(ptr, att_ptr, att.nbytes());
  }

  // 计算 attention @ V
  // 然后再直接赋值到输出的指定位置，这里也包含将多头合并到最终的输出
  vec::matmul_blas(att_ptr, total_seq_len, v_bt + h * head_dim_, v.size(-1),
                   nullptr, out_bt + h * head_dim_, out.size(-1), T /*M*/,
                   total_seq_len /*K*/, head_dim_ /*N*/);
}

void MultiHeadAttention::header_backward_native(
    const Tensor &q, const Tensor &k, const Tensor &v, dense::Tensor &grad_qkv,
    const dense::Tensor &grad_output, dense::Tensor &grad_att, size_t b,
    size_t h) {
  const auto B = grad_output.size(0);
  const auto T = grad_output.size(1);
  const auto C = grad_output.size(2);

  // 每次都要清零
  grad_att.zero_();
  auto grad_att_ptr = grad_att.mutable_data_as<float>();

  auto q_bt = q.const_data_as<float>() + b * T * q.size(-1);
  auto k_bt = k.const_data_as<float>() + b * T * k.size(-1);
  auto v_bt = v.const_data_as<float>() + b * T * v.size(-1);

  auto grad_q_bt =
      grad_qkv.mutable_data_as<float>() + b * T * grad_qkv.size(-1);
  auto grad_k_bt = grad_q_bt + C;
  auto grad_v_bt = grad_q_bt + C * 2;

  auto grad_output_bt = grad_output.const_data_as<float>() + b * T * C;

  auto att_dropout_output_bt = att_dropout_output_.const_data_as<float>() +
                               b * n_heads_ * T * T +
                               h * T * T; // 每个批次每个头的起始位置

  auto att_softmax_output_bt = att_softmax_output_.const_data_as<float>() +
                               b * n_heads_ * T * T +
                               h * T * T; // 每个批次每个头的起始位置

  // 计算 grad_att
  // grad_att = matmul(grad_input, v.transpose(-2, -1))
  vec::matmul_B_transpose_native(
      grad_output_bt + h * head_dim_, C, v_bt + h * head_dim_, v.size(-1),
      nullptr, grad_att_ptr, T, T /*M*/, T /*N*/, head_dim_ /*K*/);

  // 计算对 v 的梯度
  // grad_v = matmul(att_dropout_output_.transpose(-2, -1), grad_input)
  vec::matmul_A_transpose_native(att_dropout_output_bt, T,
                                 grad_output_bt + h * head_dim_, C, nullptr,
                                 grad_v_bt + h * head_dim_, grad_qkv.size(-1),
                                 T /*K*/, T /*M*/, head_dim_ /*N*/);

  if (attn_dropout_mask_.is_defined()) {
    auto mask_ptr = attn_dropout_mask_.const_data_as<float>() +
                    b * n_heads_ * T * T + h * T * T;

    for (size_t i = 0; i < grad_att.numel(); ++i) {
      grad_att_ptr[i] = grad_att_ptr[i] * mask_ptr[i];
    }
  }

  // softmax 反向传播
  vec::mat_softmax_backward_native(grad_att_ptr, att_softmax_output_bt,
                                   grad_att_ptr, T, T);

  if (attn_mask_.is_defined()) {
    // mask fill, att[T,T]
    auto mask_ptr = attn_mask_.const_data_as<int8_t>();
    const auto col = attn_mask_.size(-1);
    for (size_t i = 0; i < T; ++i) {
      for (size_t j = 0; j < T; ++j) {
        if (mask_ptr[i * col + j] == 1) {
          grad_att_ptr[i * T + j] = 0.0f;
        }
      }
    }
  }

  for (size_t i = 0; i < grad_att.numel(); ++i) {
    grad_att_ptr[i] *= attn_scale_;
  }
  // 计算对 q 的梯度
  // grad_q = matmul(grad_att, k)
  vec::matmul_native(grad_att_ptr, T, k_bt + h * head_dim_, k.size(-1), nullptr,
                     grad_q_bt + h * head_dim_, grad_qkv.size(-1), T /*M*/,
                     T /*K*/, head_dim_ /*N*/);

  // 计算对 k 的梯度
  // grad_k = matmul(grad_att.transpose(-2, -1), q)
  vec::matmul_A_transpose_native(grad_att_ptr, T, q_bt + h * head_dim_,
                                 q.size(-1), nullptr, grad_k_bt + h * head_dim_,
                                 grad_qkv.size(-1), T /*K*/, T /*M*/,
                                 head_dim_ /*N*/);
}

void MultiHeadAttention::header_backward_blas(const Tensor &q, const Tensor &k,
                                              const Tensor &v,
                                              dense::Tensor &grad_qkv,
                                              const dense::Tensor &grad_output,
                                              dense::Tensor &grad_att, size_t b,
                                              size_t h) {
  const auto B = grad_output.size(0);
  const auto T = grad_output.size(1);
  const auto C = grad_output.size(2);

  // 每次都要清零
  grad_att.zero_();
  auto grad_att_ptr = grad_att.mutable_data_as<float>();

  auto q_bt = q.const_data_as<float>() + b * T * q.size(-1);
  auto k_bt = k.const_data_as<float>() + b * T * k.size(-1);
  auto v_bt = v.const_data_as<float>() + b * T * v.size(-1);

  auto grad_q_bt =
      grad_qkv.mutable_data_as<float>() + b * T * grad_qkv.size(-1);
  auto grad_k_bt = grad_q_bt + C;
  auto grad_v_bt = grad_q_bt + C * 2;

  auto grad_output_bt = grad_output.const_data_as<float>() + b * T * C;

  auto att_dropout_output_bt = att_dropout_output_.mutable_data_as<float>() +
                               b * n_heads_ * T * T +
                               h * T * T; // 每个批次每个头的起始位置

  auto att_softmax_output_bt = att_softmax_output_.mutable_data_as<float>() +
                               b * n_heads_ * T * T +
                               h * T * T; // 每个批次每个头的起始位置

  // 计算 grad_att
  // grad_att = matmul(grad_input, v.transpose(-2, -1))
  vec::matmul_B_transpose_blas(
      grad_output_bt + h * head_dim_, C, v_bt + h * head_dim_, v.size(-1),
      nullptr, grad_att_ptr, T, T /*M*/, T /*N*/, head_dim_ /*K*/);

  // 计算对 v 的梯度
  // grad_v = matmul(cached_att_after_dropout.transpose(-2, -1), grad_input)
  vec::matmul_A_transpose_blas(att_dropout_output_bt, T,
                               grad_output_bt + h * head_dim_, C, nullptr,
                               grad_v_bt + h * head_dim_, grad_qkv.size(-1),
                               T /*K*/, T /*M*/, head_dim_ /*N*/);
  if (attn_dropout_mask_.is_defined()) {
    auto mask_ptr = attn_dropout_mask_.const_data_as<float>() +
                    b * n_heads_ * T * T + h * T * T;

    vec::shdm_blas(grad_att.numel(), grad_att_ptr, mask_ptr, grad_att_ptr);
  }

  // softmax 反向传播
  vec::mat_softmax_backward_blas(grad_att_ptr, att_softmax_output_bt,
                                 grad_att_ptr, T, T);

  if (attn_mask_.is_defined()) {
    // mask fill, att[T,T]
    auto mask_ptr = attn_mask_.const_data_as<int8_t>();
    const auto col = attn_mask_.size(-1);
    for (size_t i = 0; i < T; ++i) {
      for (size_t j = 0; j < T; ++j) {
        if (mask_ptr[i * col + j] == 1) {
          grad_att_ptr[i * T + j] = 0.0f;
        }
      }
    }
  }

  vec::sscal_blas(grad_att.numel(), attn_scale_, grad_att_ptr, 1);

  // 计算对 q 的梯度
  // grad_q = matmul(grad_att, k)
  vec::matmul_blas(grad_att_ptr, T, k_bt + h * head_dim_, k.size(-1), nullptr,
                   grad_q_bt + h * head_dim_, grad_qkv.size(-1), T /*M*/,
                   T /*K*/, head_dim_ /*N*/);

  // 计算对 k 的梯度
  // grad_k = matmul(grad_att.transpose(-2, -1), q)
  vec::matmul_A_transpose_blas(grad_att_ptr, T, q_bt + h * head_dim_,
                               q.size(-1), nullptr, grad_k_bt + h * head_dim_,
                               grad_qkv.size(-1), T /*K*/, T /*M*/,
                               head_dim_ /*N*/);
}

} // namespace dense