#ifndef LAYER_PATCH_EMBED_H_
#define LAYER_PATCH_EMBED_H_

#include "layer/layer.h"
#include <memory>

namespace dense {

class Conv2d;
class Flatten;
class Embedding;

// PatchEmbed是Vision Transformer
// (ViT)中的核心组件，负责将图像转换为序列化的token表示

class PatchEmbed : public Layer {
public:
  PatchEmbed(Context *ctx, const std::string &name, int64_t hidden_size,
             int64_t image_size, int64_t patch_size, int64_t num_channels,
             bool bias = true);
  ~PatchEmbed() override;

  const char *type() const override { return "patch_embed"; }

  void init() override;

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  int64_t num_patches() const { return num_patches_; }

private:
  int64_t hidden_size_;
  int64_t image_size_;
  int64_t patch_size_;
  int64_t num_channels_;

  int64_t num_patches_;

  std::unique_ptr<Conv2d> conv2d_;
  std::unique_ptr<Flatten> flatten_;
  std::unique_ptr<Embedding> pos_embedding_;

  PatchEmbed(const PatchEmbed &) = delete;
  PatchEmbed &operator=(const PatchEmbed &) = delete;
};

} // namespace dense

#endif // LAYER_PATCH_EMBED_H_