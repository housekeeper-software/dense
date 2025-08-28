#ifndef LAYER_MULTI_HEAD_ATTENTION_H_
#define LAYER_MULTI_HEAD_ATTENTION_H_

#include "layer/layer.h"
#include <memory>

namespace dense {

class Linear;
class Dropout;

class LayerCache {
public:
  virtual ~LayerCache() = default;
  virtual void reset() = 0;
  virtual void update(const dense::Tensor &new_key,
                      const dense::Tensor &new_value) = 0;
  virtual const dense::Tensor &key_states() const = 0;
  virtual const dense::Tensor &value_states() const = 0;
  virtual int64_t max_token() const = 0;
};

// GPT 的多头注意力层
class MultiHeadAttention : public Layer {
public:
  MultiHeadAttention(Context *ctx, const std::string &name, int64_t n_heads,
                     int64_t emb_dim, bool bias, float drop_rate,
                     const Tensor &attn_mask = Tensor(),
                     std::shared_ptr<LayerCache> cache = nullptr);
  ~MultiHeadAttention() override;
  const char *type() const override { return "mha"; }

  void init() override;

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  void header_forward_native(const Tensor &q, const Tensor &k, const Tensor &v,
                             Tensor &out, dense::Tensor &att, size_t b,
                             size_t h);
  void header_forward_blas(const Tensor &q, const Tensor &k, const Tensor &v,
                           Tensor &out, dense::Tensor &att, size_t b, size_t h);
  void header_backward_native(const Tensor &q, const Tensor &k, const Tensor &v,
                              dense::Tensor &grad_qkv,
                              const dense::Tensor &grad_output,
                              dense::Tensor &grad_att, size_t b, size_t h);
  void header_backward_blas(const Tensor &q, const Tensor &k, const Tensor &v,
                            dense::Tensor &grad_qkv,
                            const dense::Tensor &grad_output,
                            dense::Tensor &grad_att, size_t b, size_t h);

  std::shared_ptr<LayerCache> cache_;

  int64_t head_dim_; // 每个头的维度
  int64_t n_heads_;  // 头的个数
  int64_t emb_dim_;  // 嵌入维度
  bool bias_;        // 投影层是否使用偏置
  float drop_rate_;  // 注意力 dropout

  float attn_scale_; // 缩放因子，用于缩放注意力分数

  std::unique_ptr<Linear> in_proj_;
  std::unique_ptr<Linear> out_proj_;

  dense::Tensor attn_mask_;
  dense::Tensor attn_dropout_mask_;

  dense::Tensor q_;
  dense::Tensor k_;
  dense::Tensor v_;

  dense::Tensor att_softmax_output_;
  dense::Tensor att_dropout_output_;

  MultiHeadAttention(const MultiHeadAttention &) = delete;
  MultiHeadAttention &operator=(const MultiHeadAttention &) = delete;
};

} // namespace dense

#endif // LAYER_MULTI_HEAD_ATTENTION_H_