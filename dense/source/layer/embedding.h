#ifndef LAYER_ENBEDDING_H_
#define LAYER_ENBEDDING_H_

#include "layer/layer.h"
#include <memory>
#include <optional>

namespace dense {

// GPT token 嵌入和位置嵌入
class Embedding : public Layer {
public:
  Embedding(Context *ctx, const std::string &name, int64_t num_embeddings,
            int64_t embedding_dim, int64_t padding_idx = -1);
  ~Embedding() override = default;

  const char *type() const override { return "embedding"; }

  void init() override;

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  dense::Tensor input_cache_;
  int64_t num_embeddings_;
  int64_t embedding_dim_;
  int64_t padding_idx_;

  Embedding(const Embedding &) = delete;
  Embedding &operator=(const Embedding &) = delete;
};

} // namespace dense

#endif // LAYER_ENBEDDING_H_