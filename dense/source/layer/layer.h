#ifndef LAYER_LAYER_H_
#define LAYER_LAYER_H_

#include "base/device.h"
#include "base/tensor.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dense {

struct ParamsAndGrads {
  std::vector<Tensor> params;
  std::vector<Tensor> grads;
};

class Layer;

class Context {
public:
  Context() = default;
  ~Context() = default;
  // 注册有可学习参数的层
  void RegisterParam(Layer *layer);

  // 注册所有层
  // 注意：这里的层是指所有的层，包括有可学习参数的层
  void RegisterLayer(Layer *layer);

  Layer *GetLayer(const std::string &name);

  std::vector<Layer *> param_layers;
  std::unordered_map<std::string, Layer *> layer_map;

  bool training = false;

  Device device = DeviceType::CPU;

private:
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
};

class Layer {
public:
  Layer(Context *ctx, const std::string &name) : ctx_(ctx), name_(name) {
    ctx_->RegisterLayer(this);
  }
  virtual ~Layer() = default;
  virtual const char *type() const { return "layer"; }

  std::string name() const { return name_; }
  bool is_training() const { return ctx_->training; }

  virtual Tensor forward(const Tensor &input) = 0;
  virtual Tensor backward(const Tensor &grad_output) { return Tensor(); }

  void get_params_and_grads(std::vector<ParamsAndGrads> &params_and_grads);

  void clear_grads();

  void RegisterParam() { ctx_->RegisterParam(this); }

  Context *ctx() const { return ctx_; }

  Tensor W_;
  Tensor b_;
  Tensor grad_W_;
  Tensor grad_b_;

  // 主要用在 BatchNorm 层
  Tensor running_mean_; // 运行均值
  Tensor running_var_;  // 运行方差

private:
  Context *ctx_;
  std::string name_;

  Layer(const Layer &) = delete;
  Layer &operator=(const Layer &) = delete;
};

template <typename T, typename... Args>
std::unique_ptr<T> MakeLayerHelper(Context *ctx, const std::string &name,
                                   Args &&...args) {
  std::vector<std::unique_ptr<Layer>> layers;
  layers.reserve(sizeof...(args));
  (layers.emplace_back(std::forward<Args>(args)), ...);
  return std::make_unique<T>(ctx, name, std::move(layers));
}

template <typename... Args>
std::string make_layer_name(const char *format, Args &&...args) {
  int size = std::snprintf(nullptr, 0, format, std::forward<Args>(args)...);
  if (size <= 0) {
    return {};
  }

  auto buf = std::make_unique<char[]>(size + 1);
  std::snprintf(buf.get(), size + 1, format, std::forward<Args>(args)...);

  return std::string(buf.get(), buf.get() + size);
}

} // namespace dense

#endif // LAYER_LAYER_H_