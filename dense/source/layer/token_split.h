#ifndef LAYER_TOKEN_SPLIT_H_
#define LAYER_TOKEN_SPLIT_H_

#include "layer/layer.h"

namespace dense {

class TokenSplit : public Layer {
public:
  TokenSplit(Context *ctx, const std::string &name);
  ~TokenSplit() override = default;
  const char *type() const override { return "token_split"; }

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

private:
  std::vector<int64_t> shape_;
  TokenSplit(const TokenSplit &) = delete;
  TokenSplit &operator=(const TokenSplit &) = delete;
};

} // namespace dense

#endif // LAYER_TOKEN_SPLIT_H_