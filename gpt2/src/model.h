#ifndef MODEL_H_
#define MODEL_H_

#include "base/safe_tensors.h"
#include "kv_cache.h"
#include "layer/layer.h"
#include <functional>

namespace dense {
class Embedding;
class Dropout;
class LayerNorm;
class Linear;
class Sequential;
} // namespace dense

class SamplingChain;

class ModelConfig {
public:
  ModelConfig();
  ~ModelConfig();
  ModelConfig(const ModelConfig &);
  ModelConfig &operator=(const ModelConfig &);
  bool InitFromFile(const std::string &config_file);
  int64_t vocab_size;
  int64_t context_length;
  int64_t emb_dim;
  int64_t n_heads;
  int64_t n_layers;
  float drop_rate;
  bool qkv_bias;
  float expansion_ratio;
  float ln_epsilon;
  float initializer_range;
};

class GPTModel {
public:
  GPTModel(const ModelConfig &config, bool enable_cache = false);
  ~GPTModel();

  dense::Context *ctx();
  void from_pretrained(const std::string &filename);

  void save(const std::string &filename);

  bool is_enable_cache() const;

  void enable_training(bool enable);

  dense::Tensor forward(const dense::Tensor &input);

  std::vector<int> inference(std::vector<int> tokens, int max_length,
                             SamplingChain *chain,
                             std::function<bool(int)> token_callback = nullptr);

  void
  get_params_and_grads(std::vector<dense::ParamsAndGrads> &params_and_grads);

  void clear_grads();

  dense::Tensor backward(const dense::Tensor &grad_output);

private:
  void _load_weights();

  static size_t _write_tensor(dense::ModelParams &model_params,
                              const std::string &name,
                              const dense::Tensor &tensor);
  dense::Tensor shared_weight_;
  std::unique_ptr<dense::Embedding> wte_;
  std::unique_ptr<dense::Embedding> wpe_;
  std::unique_ptr<dense::Dropout> dropout_;

  std::vector<std::unique_ptr<dense::Sequential>> blocks_;

  // 最终的 LayerNorm
  std::unique_ptr<dense::LayerNorm> ln_f_;
  // 语言模型头
  std::unique_ptr<dense::Linear> lm_head_;

  DynamicCache cache_;

  ModelConfig config_;
  dense::ModelParams model_params_;
  dense::Context ctx_;
};
#endif // MODEL_H_