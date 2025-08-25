#include "model.h"
#include "base/storage.h"
#include <iomanip>
#include <iostream>

dense::Context *CnnModel::ctx() { return &ctx_; }

void CnnModel::AddLayer(std::unique_ptr<dense::Layer> layer) {
  layers_.emplace_back(std::move(layer));
}

void CnnModel::init_for_traning() {
  for (auto &i : ctx()->param_layers) {
    i->init();
  }
}

void CnnModel::from_pretrained(const std::string &filename) {
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

dense::Layer *CnnModel::find_layer(const std::string &name) const {
  for (const auto &i : layers_) {
    if (i->name() == name)
      return i.get();
  }
  return nullptr;
}

void CnnModel::_load_weights() {
  for (const auto &i : model_params_.tensors) {
    auto v = split(i.first, '.');
    if (v.size() < 2)
      continue;
    ;
    auto layer = find_layer(v[0]);
    if (!layer)
      continue;
    if (v[1] == "weight") {
      layer->W_ = CreateTensor(i.second);
    } else if (v[1] == "bias") {
      layer->b_ = CreateTensor(i.second);
    } else if (v[1] == "running_mean") {
      layer->running_mean_ = CreateTensor(i.second);
    } else if (v[1] == "running_var") {
      layer->running_var_ = CreateTensor(i.second);
    }
  }
}

size_t CnnModel::_write_tensor(dense::ModelParams &model_params,
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

void CnnModel::clear_grads() {
  for (auto &layer : ctx_.param_layers) {
    layer->clear_grads();
  }
}

void CnnModel::get_params_and_grads(
    std::vector<dense::ParamsAndGrads> &params_and_grads) {
  for (auto &layer : ctx_.param_layers) {
    layer->get_params_and_grads(params_and_grads);
  }
}

void CnnModel::save(const std::string &filename) {
  dense::ModelParams model_params;
  model_params.meta_data = model_params_.meta_data;
  size_t total_size = 0;
  for (const auto &i : ctx_.param_layers) {
    auto instance_name = i->name();

    if (i->W_.is_defined()) {
      total_size +=
          _write_tensor(model_params, instance_name + ".weight", i->W_);
    }
    if (i->b_.is_defined()) {
      total_size += _write_tensor(model_params, instance_name + ".bias", i->b_);
    }
    if (i->running_mean_.is_defined()) {
      total_size += _write_tensor(model_params, instance_name + ".running_mean",
                                  i->running_mean_);
    }
    if (i->running_var_.is_defined()) {
      total_size += _write_tensor(model_params, instance_name + ".running_var",
                                  i->running_var_);
    }
  }
  std::cout << "模型参数总大小: " << total_size << " bytes" << std::endl;
  model_params.save(filename);
}

void CnnModel::enable_training(bool enable) { ctx_.training = enable; }

dense::Tensor CnnModel::forward(const dense::Tensor &input) {
  dense::Tensor x = input;
  for (auto &i : layers_) {
    x = i->forward(x);
  }
  return x;
}

dense::Tensor CnnModel::backward(const dense::Tensor &grad_output) {
  dense::Tensor grad_input = grad_output;
  for (auto rit = layers_.rbegin(); rit != layers_.rend(); ++rit) {
    grad_input = (*rit)->backward(grad_input);
  }
  return grad_input;
}