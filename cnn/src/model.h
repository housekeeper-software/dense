#ifndef MODEL_H_
#define MODEL_H_

#include "base/safe_tensors.h"
#include "layer/layer.h"
#include <vector>

class CnnModel {
public:
  CnnModel() = default;
  ~CnnModel() = default;

  dense::Context *ctx();

  void AddLayer(std::unique_ptr<dense::Layer> layer);

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

  dense::Layer *find_layer(const std::string &name) const;

  static size_t _write_tensor(dense::ModelParams &model_params,
                              const std::string &name,
                              const dense::Tensor &tensor);

  std::vector<int64_t> last_conv_shape_;
  dense::ModelParams model_params_;
  dense::Context ctx_;
  std::vector<std::unique_ptr<dense::Layer>> layers_;
};
#endif // MODEL_H_