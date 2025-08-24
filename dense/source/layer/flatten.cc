#include "layer/flatten.h"
#include "layer/init.h"

namespace dense {

Flatten::Flatten(Context *ctx, const std::string &name, int64_t start_axis,
                 int64_t end_axis)
    : Layer(ctx, name), start_axis_(start_axis), end_axis_(end_axis) {}

dense::Tensor Flatten::forward(const dense::Tensor &input) {
  shape_ = input.sizes();
  const int64_t start_axis = input.canonical_dim_index(start_axis_);
  const int64_t end_axis = input.canonical_dim_index(end_axis_);
  std::vector<int64_t> new_shape;
  for (int64_t i = 0; i < start_axis; ++i) {
    new_shape.push_back(input.size(i));
  }
  const auto flattened_dim = input.count(start_axis, end_axis + 1);
  new_shape.push_back(flattened_dim);
  for (int64_t i = end_axis + 1; i < input.dim(); ++i) {
    new_shape.push_back(input.size(i));
  }
  return input.clone().reshape(new_shape);
}

dense::Tensor Flatten::backward(const dense::Tensor &grad_output) {
  return grad_output.clone().reshape(shape_);
}

} // namespace dense