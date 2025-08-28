#include "model.h"
#include "base/storage.h"
#include "layer/conv2d.h"
#include "layer/drop_path.h"
#include "layer/dropout.h"
#include "layer/embedding.h"
#include "layer/flatten.h"
#include "layer/gelu.h"
#include "layer/init.h"
#include "layer/layer_norm.h"
#include "layer/linear.h"
#include "layer/multi_head_attention.h"
#include "layer/patch_embed.h"
#include "layer/pooling.h"
#include "layer/relu.h"
#include "layer/residual.h"
#include "layer/sequential.h"
#include "layer/softmax.h"
#include "layer/token_split.h"
#include "loss/cross_entropy_loss.h"
#include "math/vec_math.h"
#include <iomanip>
#include <iostream>

VitModel::VitModel(int64_t hidden_size, int64_t image_size, int64_t patch_size,
                   int64_t num_channels) {
  ctx_.device = dense::Device(dense::DeviceType::BLAS);

  int64_t emb_dim = hidden_size;
  num_patches_ = (image_size / patch_size) * (image_size / patch_size);

  patch_embed_ = std::make_unique<dense::PatchEmbed>(
      &ctx_, "patch_embed", emb_dim, image_size, patch_size, num_channels);

  dropout_ = std::make_unique<dense::Dropout>(&ctx_, "dropout", 0.1);

  std::vector<int64_t> normalized_shape = {emb_dim};

  // 创建 n_layers 个 Block 实例
  for (size_t i = 0; i < 12; ++i) {

    auto out_dim = static_cast<int64_t>(4.0 * emb_dim);

    auto mlp = dense::MakeLayerHelper<dense::Sequential>(
        &ctx_, dense::make_layer_name("h_%d.mlp", i),
        std::make_unique<dense::Linear>(
            &ctx_, dense::make_layer_name("h_%d.mlp.c_fc", i), emb_dim,
            out_dim),
        std::make_unique<dense::GeLU>(
            &ctx_, dense::make_layer_name("h_%d.mlp.gelu", i)),
        std::make_unique<dense::Linear>(
            &ctx_, dense::make_layer_name("h_%d.mlp.c_proj", i), out_dim,
            emb_dim),
        std::make_unique<dense::DropPath>(
            &ctx_, dense::make_layer_name("h_%d.mlp.dropout", i), 0.1));

    auto residual_1 = dense::MakeLayerHelper<dense::Residual>(
        &ctx_, dense::make_layer_name("h_%d.residual_1", i),
        std::make_unique<dense::LayerNorm>(
            &ctx_, dense::make_layer_name("h_%d.ln_1", i), normalized_shape,
            1e-5, true, true),
        std::make_unique<dense::MultiHeadAttention>(
            &ctx_, dense::make_layer_name("h_%d.attn", i), 12, emb_dim, 3, true,
            0.1, false, nullptr),
        std::make_unique<dense::DropPath>(
            &ctx_, dense::make_layer_name("h_%d.attn.dropout", i), 0.1));

    auto residual_2 = dense::MakeLayerHelper<dense::Residual>(
        &ctx_, dense::make_layer_name("h_%d.residual_2", i),
        std::make_unique<dense::LayerNorm>(
            &ctx_, dense::make_layer_name("h_%d.ln_2", i), normalized_shape,
            1e-5, true, true),
        std::move(mlp));

    auto block = dense::MakeLayerHelper<dense::Sequential>(
        &ctx_, dense::make_layer_name("h_%d.block", i), std::move(residual_1),
        std::move(residual_2));

    blocks_.emplace_back(std::move(block));
  }

  ln_f_ = std::make_unique<dense::LayerNorm>(&ctx_, "ln_f", normalized_shape,
                                             1e-5, true, true);
  lm_head_ =
      std::make_unique<dense::Linear>(&ctx_, "lm_head", emb_dim, 10, false);

  token_split_ = std::make_unique<dense::TokenSplit>(&ctx_, "token_split");
}

VitModel::~VitModel() = default;

dense::Context *VitModel::ctx() { return &ctx_; }

void VitModel::init_for_traning() {
  // 参考 transformers，我们对权重做一些特殊初始化
  float initializer_range = 0.02;
  int64_t n_layers = 12;
  for (auto &i : ctx()->param_layers) {
    i->init();

    if (i->type() == "linear") {
      if (i->W_.is_defined()) {
        dense::init::normal_(i->W_, 0.0, initializer_range);
      }
      if (i->b_.is_defined()) {
        i->b_.zero_();
      }
    } else if (i->type() == "embedding") {
      if (i->W_.is_defined()) {
        dense::init::normal_(i->W_, 0.0f, initializer_range);
      }
    }
  }

  for (auto &i : ctx()->param_layers) {
    const auto name = i->name();
    if (name.length() >= 6 && name.substr(name.length() - 6) == "c_proj") {
      if (i->W_.is_defined()) {
        dense::init::normal_(i->W_, 0.0,
                             initializer_range / std::sqrt(2 * n_layers));
      }
    }
  }
}

void VitModel::from_pretrained(const std::string &filename) {
  if (!filename.empty()) {
    if (!dense::ModelParams::load(filename, &model_params_)) {
      throw std::runtime_error("加载预训练权重文件失败");
    }
    _load_weights();
  }
}

dense::Tensor CreateTensor(const dense::TensorInfo &info) {
  auto tensor = dense::Tensor::from_blob(
      dense::Tensor::dtype_from_string(info.dtype), info.shape, info.data_ptr);
  return tensor;
}

std::vector<std::string> split(const std::string &str, char delimiter) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = str.find(delimiter);

  while (end != std::string::npos) {
    std::string token = str.substr(start, end - start);
    if (!token.empty()) {
      result.push_back(token);
    }
    start = end + 1;
    end = str.find(delimiter, start);
  }

  // 处理最后一个部分
  std::string lastToken = str.substr(start);
  if (!lastToken.empty()) {
    result.push_back(lastToken);
  }

  return result;
}

void VitModel::_load_weights() {
  auto params = model_params_.tensors;

  {
    auto wpe = ctx_.GetLayer("path_embed.wpe");
    wpe->W_ = CreateTensor(params.at("path_embed.wpe.weight"));
    auto conv2d = ctx_.GetLayer("patch_embed.conv2d");
    conv2d->W_ = CreateTensor(params.at("path_embed.conv2d.weight"));
    conv2d->b_ = CreateTensor(params.at("path_embed.conv2d.bias"));
    patch_embed_->W_ = CreateTensor(params.at("path_embed.weight"));
  }

  for (size_t i = 0; i < blocks_.size(); ++i) {
    auto &block = blocks_[i];
    std::string prefix = "h." + std::to_string(i) + ".";
    {
      // 第一个 LayerNorma 层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.ln_1", i));
      layer->W_ = CreateTensor(params.at(prefix + "ln_1.weight")); // gamma
      layer->b_ = CreateTensor(params.at(prefix + "ln_1.bias"));   // beta
    }
    {
      // 多头注意力的前面线性层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.attn.c_attn", i));
      layer->W_ = CreateTensor(params.at(prefix + "attn.c_attn.weight"))
                      .transpose(0, 1);
      layer->b_ = CreateTensor(params.at(prefix + "attn.c_attn.bias"));
    }
    {
      // 多头注意力的末尾线性层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.attn.c_proj", i));
      layer->W_ = CreateTensor(params.at(prefix + "attn.c_proj.weight"))
                      .transpose(0, 1);
      layer->b_ = CreateTensor(params.at(prefix + "attn.c_proj.bias"));
    }

    {
      // 第二个 LayerNorm 层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.ln_2", i));
      layer->W_ = CreateTensor(params.at(prefix + "ln_2.weight"));
      layer->b_ = CreateTensor(params.at(prefix + "ln_2.bias"));
    }

    {
      // MLP 的第一个线性层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.mlp.c_fc", i));
      layer->W_ =
          CreateTensor(params.at(prefix + "mlp.c_fc.weight")).transpose(0, 1);
      layer->b_ = CreateTensor(params.at(prefix + "mlp.c_fc.bias"));
    }
    {
      // MLP 的第二个线性层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.mlp.c_proj", i));
      layer->W_ =
          CreateTensor(params.at(prefix + "mlp.c_proj.weight")).transpose(0, 1);
      layer->b_ = CreateTensor(params.at(prefix + "mlp.c_proj.bias"));
    }
  }
  ln_f_->W_ = CreateTensor(params.at("ln_f.weight"));
  ln_f_->b_ = CreateTensor(params.at("ln_f.bias"));
  lm_head_->W_ = CreateTensor(params.at("lm_head.weight"));
}

size_t VitModel::_write_tensor(dense::ModelParams &model_params,
                               const std::string &name,
                               const dense::Tensor &tensor) {
  if (tensor.numel() == 0) {
    std::cerr << "Warning: Tensor '" << name << "' is null, skipping write."
              << std::endl;
    return 0;
  }
  dense::TensorInfo info;
  info.storage = std::make_shared<dense::Storage>(tensor.nbytes());
  info.dtype = dense::Tensor::dtype_to_string(tensor.dtype());
  info.shape = tensor.sizes();
  info.data_ptr = reinterpret_cast<uint8_t *>(info.storage->data());
  info.data_size = tensor.nbytes();
  std::memcpy(info.storage->data(), tensor.const_data_ptr(), tensor.nbytes());
  model_params.tensors[name] = info;
  return info.data_size;
}

void VitModel::clear_grads() {
  for (auto &layer : ctx_.param_layers) {
    layer->clear_grads();
  }
}

void VitModel::get_params_and_grads(
    std::vector<dense::ParamsAndGrads> &params_and_grads) {
  for (auto &layer : ctx_.param_layers) {
    layer->get_params_and_grads(params_and_grads);
  }
}

void VitModel::save(const std::string &filename) {
  dense::ModelParams model_params;
  model_params.meta_data = model_params_.meta_data;
  size_t total_size = 0;
  {
    auto wpe = ctx_.GetLayer("path_embed.wpe");
    total_size += _write_tensor(model_params, "path_embed.wpe.weight", wpe->W_);

    auto conv2d = ctx_.GetLayer("patch_embed.conv2d");
    total_size +=
        _write_tensor(model_params, "path_embed.conv2d.weight", conv2d->W_);
    total_size +=
        _write_tensor(model_params, "path_embed.conv2d.bias", conv2d->b_);

    total_size +=
        _write_tensor(model_params, "path_embed.weight", patch_embed_->W_);
  }

  for (size_t i = 0; i < blocks_.size(); ++i) {
    auto &block = blocks_[i];
    std::string prefix = "h." + std::to_string(i) + ".";
    {
      // 第一个 LayerNorm 层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.ln_1", i));
      total_size +=
          _write_tensor(model_params, prefix + "ln_1.weight", layer->W_);
      total_size +=
          _write_tensor(model_params, prefix + "ln_1.bias", layer->b_);
    }
    {
      // 多头注意力的起始线性层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.attn.c_attn", i));
      total_size += _write_tensor(model_params, prefix + "attn.c_attn.weight",
                                  layer->W_.clone().transpose(0, 1));
      total_size +=
          _write_tensor(model_params, prefix + "attn.c_attn.bias", layer->b_);
    }

    {
      // 多头注意力后面的投影层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.attn.c_proj", i));
      total_size += _write_tensor(model_params, prefix + "attn.c_proj.weight",
                                  layer->W_.clone().transpose(0, 1));
      total_size +=
          _write_tensor(model_params, prefix + "attn.c_proj.bias", layer->b_);
    }
    {
      // 第二个 LayerNorm 层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.ln_2", i));
      total_size +=
          _write_tensor(model_params, prefix + "ln_2.weight", layer->W_);
      total_size +=
          _write_tensor(model_params, prefix + "ln_2.bias", layer->b_);
    }
    {
      // MLP 的第一个线性层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.mlp.c_fc", i));
      total_size += _write_tensor(model_params, prefix + "mlp.c_fc.weight",
                                  layer->W_.clone().transpose(0, 1));
      total_size +=
          _write_tensor(model_params, prefix + "mlp.c_fc.bias", layer->b_);
    }
    {
      // MLP 第二个投影层
      auto layer = ctx_.GetLayer(dense::make_layer_name("h_%d.mlp.c_proj", i));
      total_size += _write_tensor(model_params, prefix + "mlp.c_proj.weight",
                                  layer->W_.clone().transpose(0, 1));
      total_size +=
          _write_tensor(model_params, prefix + "mlp.c_proj.bias", layer->b_);
    }
  }
  total_size += _write_tensor(model_params, "ln_f.weight", ln_f_->W_);
  total_size += _write_tensor(model_params, "ln_f.bias", ln_f_->b_);
  total_size += _write_tensor(model_params, "lm_head.weight", lm_head_->W_);

  std::cout << "模型参数总大小: " << total_size << " bytes" << std::endl;
  model_params.save(filename);
}

void VitModel::enable_training(bool enable) { ctx_.training = enable; }

dense::Tensor VitModel::forward(const dense::Tensor &input) {
  dense::Tensor x = patch_embed_->forward(input);
  x = dropout_->forward(x);

  for (size_t i = 0; i < blocks_.size(); ++i) {
    auto &block = blocks_[i];
    x = block->forward(x);
  }
  x = ln_f_->forward(x);
  x = lm_head_->forward(x);
  return token_split_->forward(x);
}

dense::Tensor VitModel::backward(const dense::Tensor &grad_output) {
  auto x = token_split_->backward(grad_output);
  x = lm_head_->backward(x);
  x = ln_f_->backward(x);

  // 3. Transformer Block 层 `h_` 的反向传播（逆序遍历）
  // 从最后一个 Block 开始，到第一个 Block
  for (int64_t i = blocks_.size() - 1; i >= 0; --i) {
    auto &block = blocks_[i];
    // 每个 block 的 backward 接收来自上一个 block（或 ln_f_）的梯度，并返回对该
    // block 输入的梯度
    x = block->backward(x);
  }

  // grad 此时是损失对 (tok_emb + pos_emb) 之后，dropout 之前的输出的梯度
  x = dropout_->backward(x);
  return patch_embed_->backward(x);
}
