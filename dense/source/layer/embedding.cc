#include "layer/embedding.h"
#include "layer/init.h"
#include "math/vec_math.h"
#include <stdexcept>

namespace dense {

Embedding::Embedding(Context *ctx, const std::string &name,
                     int64_t num_embeddings, int64_t embedding_dim,
                     int64_t padding_idx)
    : Layer(ctx, name), num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim), padding_idx_(padding_idx) {
  // 这层有可学习参数 W_,嵌入权重
  RegisterParam();
}

void Embedding::init() {
  std::vector<int64_t> w_shape = {num_embeddings_, embedding_dim_};

  W_ = Tensor::empty(DType::kFloat32, w_shape);
  init::normal_(W_);
  if (padding_idx_ >= 0) {
    auto ptr = W_.mutable_data_as<float>() + padding_idx_ * embedding_dim_;
    std::fill_n(ptr, embedding_dim_, 0.0f);
  }
}

dense::Tensor Embedding::forward(const dense::Tensor &input) {
  if (input.dim() < 1) {
    throw std::runtime_error("输入张量至少需要1维");
  }
  if (input.dtype() != dense::DType::kInt64) {
    throw std::runtime_error("输入张量的类型必须是:kInt64");
  }

  if (is_training()) {
    input_cache_ = input.clone();
  }

  auto input_shape = input.sizes();

  std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
  output_shape.push_back(embedding_dim_);

  auto output = Tensor::zeros(DType::kFloat32, output_shape);

  const int64_t total_elements = input.numel();

  auto flat_input = input.reshape({total_elements});
  auto flat_output = output.reshape({total_elements, embedding_dim_});

  auto w_ptr = W_.const_data_as<float>();
  auto output_ptr = flat_output.mutable_data_as<float>();
  auto input_ptr = flat_input.const_data_as<int64_t>();

  for (size_t i = 0; i < total_elements; ++i) {
    auto idx = input_ptr[i];

    if (idx < 0 || idx >= num_embeddings_) {
      throw std::runtime_error("索引超出范围: " + std::to_string(idx) +
                               ", 有效范围: [0, " +
                               std::to_string(num_embeddings_) + ")");
    }
    auto out_ptr = output_ptr + i * embedding_dim_;
    auto w_idx_ptr = w_ptr + idx * embedding_dim_;

    // 复制嵌入向量
    if (ctx()->device.is_blas()) {
      vec::scopy_blas(embedding_dim_, w_idx_ptr, 1, out_ptr, 1);
    } else {
      std::copy_n(w_idx_ptr, embedding_dim_, out_ptr);
    }
  }
  return output;
}

dense::Tensor Embedding::backward(const dense::Tensor &grad_output) {
  if (!grad_W_.is_defined()) {
    grad_W_ = dense::Tensor::zeros_like(W_);
  }
  auto input = input_cache_;

  const int64_t total_elements = input.numel();

  // 将梯度输出张量展平为二维张量
  auto flat_grad_output = grad_output.reshape({total_elements, embedding_dim_});
  auto flat_input = input.reshape({total_elements});

  auto grad_output_ptr = flat_grad_output.const_data_as<float>();
  auto input_ptr = flat_input.const_data_as<int64_t>();
  auto grad_w_ptr = grad_W_.mutable_data_as<float>();

  for (size_t i = 0; i < total_elements; ++i) {
    auto idx = input_ptr[i];

    if (idx == padding_idx_) {
      // 如果是填充索引，则跳过梯度计算和累加
      continue;
    }

    auto grad_out_ptr = grad_output_ptr + i * embedding_dim_;
    auto grad_w_idx_ptr = grad_w_ptr + idx * embedding_dim_;
    // 将梯度累加到嵌入矩阵的这个特征向量上
    if (ctx()->device.is_blas()) {
      vec::saxpy_blas(embedding_dim_, 1.0f, grad_out_ptr, 1, grad_w_idx_ptr, 1);
    } else {
      for (int64_t k = 0; k < num_embeddings_; ++k) {
        // 因为输入中可能包含多个相同的子词，所以要累加!!!
        grad_w_idx_ptr[k] += grad_out_ptr[k];
      }
    }
  }
  // 返回空张量，因为嵌入层没有输入梯度
  // 在实际应用中，嵌入层通常不需要返回输入梯度
  return dense::Tensor();
}

} // namespace dense