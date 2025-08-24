#include "layer/layer.h"

namespace dense {

void Context::RegisterParam(Layer *layer) { param_layers.emplace_back(layer); }

void Context::RegisterLayer(Layer *layer) {
  layer_map.insert({layer->name(), layer});
}

Layer *Context::GetLayer(const std::string &name) {
  auto it = layer_map.find(name);
  if (it != layer_map.end()) {
    return it->second;
  }
  return nullptr;
}

void Layer::get_params_and_grads(
    std::vector<ParamsAndGrads> &params_and_grads) {
  ParamsAndGrads params;
  if (W_.is_defined()) {
    params.params.emplace_back(W_);
  }
  if (b_.is_defined()) {
    params.params.emplace_back(b_);
  }
  if (grad_W_.is_defined()) {
    params.grads.emplace_back(grad_W_);
  }
  if (grad_b_.is_defined()) {
    params.grads.emplace_back(grad_b_);
  }
  if (!params.grads.empty()) {
    params_and_grads.emplace_back(params);
  }
}

void Layer::clear_grads() {
  if (grad_W_.is_defined()) {
    grad_W_.zero_();
  }
  if (grad_b_.is_defined()) {
    grad_b_.zero_();
  }
}

} // namespace dense