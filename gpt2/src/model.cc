#include "model.h"
#include "base/storage.h"
#include "json.hpp"
#include "layer/dropout.h"
#include "layer/embedding.h"
#include "layer/gelu.h"
#include "layer/layer_norm.h"
#include "layer/linear.h"
#include "layer/residual.h"
#include "layer/sequential.h"
#include "sampling.h"
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>

ModelConfig::ModelConfig()
    : vocab_size(50257), context_length(1024), emb_dim(768), n_heads(12),
      n_layers(12), drop_rate(0.1), qkv_bias(true), expansion_ratio(4.0f),
      ln_epsilon(1e-05) {}
ModelConfig::~ModelConfig() = default;
ModelConfig::ModelConfig(const ModelConfig &) = default;
ModelConfig &ModelConfig::operator=(const ModelConfig &) = default;

bool ModelConfig::InitFromFile(const std::string &config_file) {
  std::ifstream ifs(config_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open config file: " << config_file << std::endl;
    return false;
  }

  nlohmann::json config_json;
  ifs >> config_json;

  // Parse JSON and initialize model config
  try {
    vocab_size = config_json.at("vocab_size").get<int64_t>();
    context_length = config_json.at("n_ctx").get<int64_t>();
    emb_dim = config_json.at("n_embd").get<int64_t>();
    n_heads = config_json.at("n_head").get<int64_t>();
    n_layers = config_json.at("n_layer").get<int64_t>();
    drop_rate = config_json.at("attn_pdrop").get<float>();
    if (config_json.contains("layer_norm_epsilon")) {
      ln_epsilon = config_json.at("layer_norm_epsilon").get<float>();
    }
  } catch (const std::exception &e) {
    std::cerr << "Error parsing config file: " << e.what() << std::endl;
    return false;
  }

  return true;
}

GPTModel::GPTModel(const ModelConfig &config, bool enable_cache)
    : config_(config), cache_(config.n_layers, config.context_length) {
  cache_.set_enabled(enable_cache);
  ctx_.device = dense::Device(dense::DeviceType::BLAS);

  wte_ = std::make_unique<dense::Embedding>(
      &ctx_, "wte.weight", config_.vocab_size, config_.emb_dim, 50256);
  wpe_ = std::make_unique<dense::Embedding>(
      &ctx_, "wpe.weight", config_.context_length, config_.emb_dim);
  dropout_ = std::make_unique<dense::Dropout>(&ctx_, "dropout",
                                              config_.drop_rate /*0.1*/);

  std::vector<int64_t> normalized_shape = {config_.emb_dim};

  // 创建 n_layers 个 Block 实例
  for (size_t i = 0; i < config_.n_layers; ++i) {

    auto out_dim =
        static_cast<int64_t>(config_.expansion_ratio * config_.emb_dim);

    auto mlp = dense::MakeLayerHelper<dense::Sequential>(
        &ctx_, dense::make_layer_name("h_%d.mlp", i),
        std::make_unique<dense::Linear>(
            &ctx_, dense::make_layer_name("h_%d.mlp.c_fc", i), config_.emb_dim,
            out_dim),
        std::make_unique<dense::GeLU>(
            &ctx_, dense::make_layer_name("h_%d.mlp.gelu", i)),
        std::make_unique<dense::Linear>(
            &ctx_, dense::make_layer_name("h_%d.mlp.c_proj", i), out_dim,
            config_.emb_dim),
        std::make_unique<dense::Dropout>(
            &ctx_, dense::make_layer_name("h_%d.mlp.dropout", i),
            config_.drop_rate));

    auto residual_1 = dense::MakeLayerHelper<dense::Residual>(
        &ctx_, dense::make_layer_name("h_%d.residual_1", i),
        std::make_unique<dense::LayerNorm>(
            &ctx_, dense::make_layer_name("h_%d.ln_1", i), normalized_shape,
            config_.ln_epsilon, config_.qkv_bias),
        std::make_unique<dense::MultiHeadAttention>(
            &ctx_, dense::make_layer_name("h_%d.attn", i), config_.n_heads,
            config_.emb_dim, config_.context_length, config_.qkv_bias,
            config_.drop_rate, cache_.get(i)),
        std::make_unique<dense::Dropout>(
            &ctx_, dense::make_layer_name("h_%d.attn.dropout", i),
            config_.drop_rate));

    auto residual_2 = dense::MakeLayerHelper<dense::Residual>(
        &ctx_, dense::make_layer_name("h_%d.residual_2", i),
        std::make_unique<dense::LayerNorm>(
            &ctx_, dense::make_layer_name("h_%d.ln_2", i), normalized_shape,
            config_.ln_epsilon, config_.qkv_bias),
        std::move(mlp));

    auto block = dense::MakeLayerHelper<dense::Sequential>(
        &ctx_, dense::make_layer_name("h_%d.block", i), std::move(residual_1),
        std::move(residual_2));

    blocks_.emplace_back(std::move(block));
  }

  ln_f_ = std::make_unique<dense::LayerNorm>(&ctx_, "ln_f", normalized_shape,
                                             config_.ln_epsilon, true);
  lm_head_ = std::make_unique<dense::Linear>(&ctx_, "lm_head", config_.emb_dim,
                                             config_.vocab_size, false);
  // lm_head 于 wte 共享权重
  lm_head_->W_ = wte_->W_;
}

GPTModel::~GPTModel() = default;

dense::Context *GPTModel::ctx() { return &ctx_; }

void GPTModel::from_pretrained(const std::string &filename) {
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

void GPTModel::_load_weights() {
  auto params = model_params_.tensors;
  auto wte_weight = CreateTensor(params.at("wte.weight"));

  wte_->W_ = wte_weight;
  lm_head_->W_ = wte_weight;

  wpe_->W_ = CreateTensor(params.at("wpe.weight"));

  ln_f_->W_ = CreateTensor(params.at("ln_f.weight"));
  ln_f_->b_ = CreateTensor(params.at("ln_f.bias"));

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
}

size_t GPTModel::_write_tensor(dense::ModelParams &model_params,
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

void GPTModel::clear_grads() {
  for (auto &layer : ctx_.param_layers) {
    layer->clear_grads();
  }
}

void GPTModel::get_params_and_grads(
    std::vector<dense::ParamsAndGrads> &params_and_grads) {
  for (auto &layer : ctx_.param_layers) {
    layer->get_params_and_grads(params_and_grads);
  }
}

bool GPTModel::is_enable_cache() const {
  return cache_.enabled() && !ctx_.training;
}

void GPTModel::save(const std::string &filename) {
  dense::ModelParams model_params;
  model_params.meta_data = model_params_.meta_data;
  size_t total_size = 0;
  total_size += _write_tensor(model_params, "wte.weight", wte_->W_);
  total_size += _write_tensor(model_params, "wpe.weight", wpe_->W_);
  total_size += _write_tensor(model_params, "ln_f.weight", ln_f_->W_);
  total_size += _write_tensor(model_params, "ln_f.bias", ln_f_->b_);

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
  std::cout << "模型参数总大小: " << total_size << " bytes" << std::endl;
  model_params.save(filename);
}

void GPTModel::enable_training(bool enable) { ctx_.training = enable; }

dense::Tensor GPTModel::forward(const dense::Tensor &input) {
  auto B = input.size(0);
  auto T = input.size(1);
  assert(T <= config_.context_length);

  std::vector<int64_t> pos_data(B * T);
  size_t current_pos = 0;
  if (is_enable_cache()) {
    current_pos = cache_.get_seq_length();
  }
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      pos_data[b * T + t] = current_pos + static_cast<int64_t>(t);
    }
  }

  auto pos =
      dense::Tensor::from_blob(dense::DType::kInt64, {B, T}, &pos_data[0]);

  auto tok_emb = wte_->forward(input);
  auto pos_emb = wpe_->forward(pos);
  {
    auto N = tok_emb.numel();
    auto tok_ptr = tok_emb.mutable_data_as<float>();
    auto pos_ptr = pos_emb.const_data_as<float>();
    for (size_t i = 0; i < N; ++i) {
      tok_ptr[i] = tok_ptr[i] + pos_ptr[i];
    }
  }
  auto x = dropout_->forward(tok_emb);

  for (size_t i = 0; i < blocks_.size(); ++i) {
    auto &block = blocks_[i];
    x = block->forward(x);
  }
  x = ln_f_->forward(x);
  return lm_head_->forward(x);
}

dense::Tensor GPTModel::backward(const dense::Tensor &grad_output) {
  auto grad_input = lm_head_->backward(grad_output);
  grad_input = ln_f_->backward(grad_input);

  // 3. Transformer Block 层 `h_` 的反向传播（逆序遍历）
  // 从最后一个 Block 开始，到第一个 Block
  for (int64_t i = blocks_.size() - 1; i >= 0; --i) {
    auto &block = blocks_[i];
    // 每个 block 的 backward 接收来自上一个 block（或 ln_f_）的梯度，并返回对该
    // block 输入的梯度
    grad_input = block->backward(grad_input);
  }

  // grad 此时是损失对 (tok_emb + pos_emb) 之后，dropout 之前的输出的梯度
  grad_input = dropout_->backward(grad_input);

  // 5. 词嵌入和位置嵌入的反向传播
  // dropout_ 的输入是 tok_emb_cache_ + pos_emb_cache_
  // 因此，grad （即 dL/d_(tok_emb+pos_emb)）需要分别传递给 tok_emb 和 pos_emb
  // dL/d_tok_emb = dL/d_(tok_emb+pos_emb)
  // dL/d_pos_emb = dL/d_(tok_emb+pos_emb)

  // wte_ (Embedding) 的 backward
  // wte_ 的 backward 接收 dL/d_tok_emb
  wte_->backward(grad_input); // 这是损失对原始 token input 的梯度

  // wpe_ (Embedding) 的 backward
  // wpe_ 的 backward 接收 dL/d_pos_emb
  // 注意：pos_emb 是通过位置索引生成的，通常位置嵌入不需要计算对原始 `pos`
  // 索引的梯度， 而是直接更新 `wpe_` 自身的权重。这里调用 `wpe_->backward`
  // 即可。
  wpe_->backward(grad_input); // 对位置嵌入权重的梯度会在这里计算

  return grad_input;
}

std::vector<int> GPTModel::inference(std::vector<int> tokens, int max_length,
                                     SamplingChain *chain,
                                     std::function<bool(int)> token_callback) {
  assert(chain != nullptr);
  if (is_enable_cache() &&
      (max_length + tokens.size() > config_.context_length)) {
    std::cerr << "错误:启用缓存时,生成长度加上初始token数量不能超过模型上下文"
                 "长度。\n";
    return {};
  }

  std::vector<int> result_tokens = tokens;
  while (result_tokens.size() < max_length) {
    dense::Tensor input_tensor;
    std::vector<int64_t> input_tokens(result_tokens.begin(),
                                      result_tokens.end());
    if (is_enable_cache()) {
      if (result_tokens.size() == tokens.size()) {
        input_tensor = dense::Tensor::from_blob(
            dense::DType::kInt64,
            {1, static_cast<int64_t>(input_tokens.size())}, &input_tokens[0]);
      } else {
        input_tensor =
            dense::Tensor::from_blob(dense::DType::kInt64, {1, 1},
                                     &input_tokens[input_tokens.size() - 1]);
      }
    } else {
      input_tensor = dense::Tensor::from_blob(
          dense::DType::kInt64, {1, static_cast<int64_t>(input_tokens.size())},
          &input_tokens[0]);
    }
    auto logits = forward(input_tensor);
    // [B,T,C]
    auto B = logits.size(0);
    auto T = logits.size(1);
    auto C = logits.size(2);

    // 只取第一个批次的最后一个token预测向量
    auto ptr = logits.mutable_data_as<float>() + (T - 1) * C;

    std::vector<llama_token_data> cur;
    cur.reserve(C);
    for (llama_token token_id = 0; token_id < C; ++token_id) {
      cur.emplace_back(llama_token_data(token_id, ptr[token_id], 0.0f));
    }

    llama_token_data_array cur_p(&cur[0], cur.size());
    auto next_token_id = chain->sample(&cur_p);

    result_tokens.push_back(next_token_id);

    if (token_callback) {
      // 如果提供了回调函数，调用它来处理生成的token。
      if (!token_callback(next_token_id)) {
        break;
      }
    } else if (next_token_id == 50256) { // GPT-2的EOS token ID通常是50256
      // std::cerr << "\n生成了EOS token，提前停止生成。\n";
      break;
    }
  }

  return result_tokens;
}