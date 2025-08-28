#ifndef MODEL_H_
#define MODEL_H_

#include "base/safe_tensors.h"
#include "layer/layer.h"
#include <vector>

namespace dense {
class Conv2d;
class Embedding;
class Dropout;
class LayerNorm;
class Linear;
class Sequential;
class Flatten;
class PatchEmbed;
class TokenSplit;
} // namespace dense

class ModelConfig {
public:
  ModelConfig();
  ~ModelConfig();
  ModelConfig(const ModelConfig &);
  ModelConfig &operator=(const ModelConfig &);
  bool InitFromFile(const std::string &config_file);
  int64_t emb_dim;
  int64_t n_heads;
  int64_t n_layers;
  float drop_rate;
  bool qkv_bias;
  float expansion_ratio;
  float ln_epsilon;
  float initializer_range;
};

class VitModel {
public:
  VitModel(const ModelConfig &config, int64_t image_size, int64_t patch_size,
           int64_t num_channels, int64_t num_classes);
  ~VitModel();

  dense::Context *ctx();

  void init_for_traning();

  dense::Tensor forward(const dense::Tensor &input);
  dense::Tensor backward(const dense::Tensor &grad_output);

  void from_pretrained(const std::string &filename);
  void save(const std::string &filename);
  void enable_training(bool enable);

  void
  get_params_and_grads(std::vector<dense::ParamsAndGrads> &params_and_grads);

  void clear_grads();

private:
  void _load_weights();

  static size_t _write_tensor(dense::ModelParams &model_params,
                              const std::string &name,
                              const dense::Tensor &tensor);

  ModelConfig config_;

  std::unique_ptr<dense::PatchEmbed> patch_embed_;

  std::unique_ptr<dense::Dropout> dropout_;

  std::vector<std::unique_ptr<dense::Sequential>> blocks_;

  // 最终的 LayerNorm
  std::unique_ptr<dense::LayerNorm> ln_f_;
  // 语言模型头
  std::unique_ptr<dense::Linear> lm_head_;

  std::unique_ptr<dense::TokenSplit> token_split_;

  dense::ModelParams model_params_;
  dense::Context ctx_;
};
#endif // MODEL_H_